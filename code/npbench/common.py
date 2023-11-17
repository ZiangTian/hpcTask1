import argparse
import copy
import csv
import itertools
import numpy as np
import os
import pathlib
import sys
import timeit
import traceback
import time
from typing import Any, Dict

from dace import dtypes


# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def time_to_ms(raw):
    return int(round(raw * 1000))


def benchmark(stmt, setup="pass", out_text="Runtime", repeat=1, context=None):
    if repeat <= 1:
        if setup and setup != "pass":
            exec(setup, context)
        start = time.time()
        res = eval(stmt, context)
        finish = time.time()
        raw_time = finish - start
        raw_time_list = [raw_time]
    else:
        res = None
        raw_time_list = timeit.repeat(stmt,
                                      setup=setup,
                                      repeat=repeat,
                                      number=1,
                                      globals=context)
        raw_time = np.median(raw_time_list)
    ms_time = time_to_ms(raw_time)
    print("{}: {}ms".format(out_text, ms_time))
    return res, raw_time_list


# cupy.cuda.stream.get_current_stream().synchronize()


def benchmark_cupy(stmt,
                   setup="pass",
                   out_text="Runtime",
                   repeat=1,
                   context=None):
    import cupy as cp
    raw_time_list = []
    for _ in range(repeat):
        if setup and setup != "pass":
            exec(setup, context)
        cp.cuda.stream.get_current_stream().synchronize()
        start = time.time()
        res = eval(stmt, context)
        cp.cuda.stream.get_current_stream().synchronize()
        finish = time.time()
        raw_time = finish - start
        raw_time_list.append(raw_time)
    raw_time = np.median(raw_time_list)
    ms_time = time_to_ms(raw_time)
    print("{}: {}ms".format(out_text, ms_time))
    return res, raw_time_list


def relative_error(ref, val):
    return np.linalg.norm(ref - val) / np.linalg.norm(ref)


def validation(ref, val, framework="Unknown"):
    if not isinstance(ref, (tuple, list)):
        ref = [ref]
    if not isinstance(val, (tuple, list)):
        val = [val]
    valid = True
    for r, v in zip(ref, val):
        if not np.allclose(r, v):
            try:
                import cupy
                if isinstance(v, cupy.ndarray):
                    relerror = relative_error(r, cupy.asnumpy(v))
                else:
                    relerror = relative_error(r, v)
            except Exception as e:
                relerror = relative_error(r, v)
            if relerror < 1e-3:
                continue
            valid = False
            print("Relative error: {}".format(relerror))
            # return False
    if not valid:
        print("{} did not validate!".format(framework))
    return valid


def write_csv(file_name, field_names, values, append=False):
    write_mode = 'w'
    if append:
        write_mode = 'a'
    with open(file_name, mode=write_mode) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)

        writer.writeheader()
        for entry in values:
            writer.writerow(entry)


def run_mode(exec_str: str,
             setup_str: str,
             report_str: str,
             np_out: Any = None,
             mode: str = "main",
             validate: bool = True,
             repeat: int = 10,
             context: Dict[str, Any] = None,
             out_args=None,
             out_in_device: bool = False,
             framework: str = ""):
    if mode not in ("first", "main", "papi"):
        raise ValueError("Invalid input {} for mode keyword "
                         "argument.".format(mode))

    bench_func = benchmark
    if framework == "cupy":
        bench_func = benchmark_cupy

    # Always run "first"
    try:
        if out_args:
            ldict = {}
            exec(setup_str, {**context, **locals()}, ldict)
            ct_out, fe_time = bench_func(exec_str,
                                         setup=None,
                                         out_text=report_str +
                                         " first execution",
                                         context={
                                             **context,
                                             **ldict
                                         })
        else:
            ct_out, fe_time = bench_func(exec_str,
                                         setup=setup_str,
                                         out_text=report_str +
                                         " first execution",
                                         context=context)
        if out_args:
            ct_out = [ldict[a] for a in out_args]
        # if out_in_device:
        #     def copy_to_device(a):
        #         with cp.cuda.device(a.device):
        #             return cp.asnumpy(a)
        #     ldict = {}
        #     if isinstance(ct_out, (list, tuple)):
        #         exec("host_out = [copy_to_device(a) for a in ct_out]",
        #              {**context, **locals()}, ldict)
        #         # ct_out = [cp.asnumpy(a) for a in ct_out]
        #     else:
        #         exec("host_out = cp.asnumpy(ct_out)",
        #              {**context, **locals()}, ldict)
        #         # ct_out = cp.asnumpy(ct_out)
        #     ct_out = ldict["host_out"]

    except Exception as e:
        print("Failed to benchmark {} first execution.".format(report_str))
        print(e)
        traceback.print_exc()
        return None, None

    if mode == "first":
        return fe_time, True

    if mode == "main":
        valid = True
        if validate and np_out is not None:
            try:
                valid = validation(np_out, ct_out, report_str)
            except Exception as e:
                print("Failed to run {} validation.".format(report_str))
        try:
            _, me_time = bench_func(exec_str,
                                    setup=setup_str,
                                    out_text=report_str + " main execution",
                                    repeat=repeat,
                                    context=context)
        except Exception as e:
            print("Failed to benchmark {} main execution.".format(report_str))
            return None, None

        return me_time, valid

    if mode == "papi":
        try:
            from pypapi import papi_high
            from pypapi import events as papi_events

            events = [papi_events.PAPI_DP_OPS]
            # events = [papi_events.PAPI_L2_TCM, papi_events.PAPI_L2_TCA]
            # events = [papi_events.PAPI_L3_TCM, papi_events.PAPI_L3_TCA]

            if setup_str != "pass":
                eval(setup_str, context)
            papi_high.start_counters(events)
            eval(exec_str, globals())
            counters = papi_high.read_counters()
            papi_high.stop_counters()
        except Exception as e:
            print("Failed to benchmark {} PAPI counters .".format(report_str))
            raise e

        return None, None


def run_numba(
    module_name: str,  # Module name
    kind: str,  # Kind, e.g., (micro)benchmark or (micro)application
    domain: str,  # Scientific domain, e.g., CFDs
    dwarf: str,  # Berkeley dwarf, e.g., dense linear algebra
    finfo: Dict[str, Any],  # Framework information
    mode: str = "main",  # Benchmark mode
    validate: bool = True,  # Enables validation against NumPy,
    repeat: int = 10,  # Benchmark repetitions
    context: Dict[Any, Any] = None  # Used to pass test-specific variables
):
    benchmark_values = []

    for func_name in ("object_mode", "object_mode_parallel",
                      "object_mode_prange", "nopython_mode",
                      "nopython_mode_parallel", "nopython_mode_prange"):
        # Import Numba implementation
        try:
            exec("from {m} import {f} as ct_impl".format(m=finfo["module_str"],
                                                         f=func_name))
        except Exception as e:
            print("Failed to load Numba {} implementation.".format(func_name))
            print(e)
            continue

        exec_str = "ct_impl({a})".format(a=finfo["arg_str"])
        setup_str = finfo["setup_str"]
        report_str = "{r} {f}".format(r=finfo["report_str"], f=func_name)
        np_out = context["np_out"]

        out_args = None
        if "out_args" in finfo.keys():
            out_args = finfo["out_args"]

        values, valid = run_mode(exec_str, setup_str, report_str, np_out, mode,
                                 validate, repeat, {
                                     **context,
                                     **locals()
                                 }, out_args)

        if values:
            for vl in values:
                benchmark_values.append(
                    dict(benchmark=module_name,
                         kind=kind,
                         domain=domain,
                         dwarf=dwarf,
                         framework="Numba",
                         mode=mode,
                         details=func_name,
                         validated=valid,
                         time=vl))

    return benchmark_values


def run_dace(
    module_name: str,  # Module name
    kind: str,  # Kind, e.g., (micro)benchmark or (micro)application
    domain: str,  # Scientific domain, e.g., CFDs
    dwarf: str,  # Berkeley dwarf, e.g., dense linear algebra
    finfo: Dict[str, Any],  # Framework information
    mode: str = "main",  # Benchmark mode
    validate: bool = True,  # Enables validation against NumPy,
    repeat: int = 10,  # Benchmark repetitions
    context: Dict[Any, Any] = None  # Used to pass test-specific variables
):
    benchmark_values = []

    # Import DaCe implementation
    try:
        import dace
        import dace.data
        import dace.dtypes as dtypes
        from dace.transformation.optimizer import Optimizer
        from dace.transformation.dataflow import MapFusion, Vectorization, MapCollapse
        from dace.transformation.interstate import LoopToMap
        import dace.transformation.auto_optimize as opt
        
        exec("from {m} import {f} as ct_impl".format(m=finfo["module_str"],
                                                     f=finfo["func_str"]))
    except Exception as e:
        print("Failed to load the DaCe implementation.")
        raise (e)
    
    #########################################################
    # Prepare SDFGs
    base_sdfg, parse_time = benchmark("ct_impl.to_sdfg(strict=False)",
                                      out_text="DaCe parsing time",
                                      context={
                                          **context,
                                          **locals()
                                      })
    strict_sdfg = copy.deepcopy(base_sdfg)
    strict_sdfg._name = "strict"
    _, strict_time = benchmark("strict_sdfg.apply_strict_transformations()",
                               out_text="DaCe Strict Transformations time",
                               context={
                                   **context,
                                   **locals()
                               })
    sdfg_list = [strict_sdfg]
    time_list = [parse_time[0] + strict_time[0]]
    
    '''
    for name, value in itertools.chain(locals().items(), context.items()):
        if name in strict_sdfg.free_symbols:
            strict_sdfg.specialize({name:value})
    '''
    ##########################################################

    try:
        fusion_sdfg = copy.deepcopy(strict_sdfg)
        fusion_sdfg._name = "fusion"
        _, fusion_time = benchmark(
            "fusion_sdfg.apply_transformations_repeated([MapFusion])",
            out_text="DaCe MapFusion time",
            context={
                **context,
                **locals()
            })
        sdfg_list.append(fusion_sdfg)
        time_list.append(time_list[-1] + fusion_time[0])
    except Exception as e:
        print("DaCe MapFusion failed")
        print(e)
        fusion_sdfg = copy.deepcopy(strict_sdfg)


    ###########################################################

    def parallelize(sdfg):
        from dace.sdfg import propagation
        strict_xforms = dace.transformation.strict_transformations()

        for sd in sdfg.all_sdfgs_recursive():
            propagation.propagate_states(sd)
        sdfg.apply_transformations_repeated([LoopToMap, MapCollapse] +
                                            strict_xforms,
                                            strict=True)

    try:
        parallel_sdfg = copy.deepcopy(fusion_sdfg)
        parallel_sdfg._name = "parallel"
        _, ptime1 = benchmark("parallelize(parallel_sdfg)",
                              out_text="DaCe LoopToMap time1",
                              context={
                                  **context,
                                  **locals()
                              })
        _, ptime2 = benchmark(
            "parallel_sdfg.apply_transformations_repeated([MapFusion])",
            out_text="DaCe LoopToMap time2",
            context={
                **context,
                **locals()
            })
        sdfg_list.append(parallel_sdfg)
        time_list.append(time_list[-1] + ptime1[0] + ptime2[0])

    except Exception as e:
        print("DaCe LoopToMap failed")
        print(e)
        parallel_sdfg = copy.deepcopy(fusion_sdfg)

    ##########################################################
    '''
    def tile(sdfg):
        for graph in sdfg.nodes():
            for node in graph.nodes():
                if isinstance(node, dace.sdfg.nodes.MapEntry):
                    dace.transformation.data.flow.MapTiling.apply_to(sdfg, _map_entry = node)
        return sdfg 

    try:
        tiled_sdfg = copy.deepcopy(parallel_sdfg)
        tiled_sdfg.name = "tiled"
        _, ttime1 = benchmark("tile(tiled_sdfg)",
                              out_text="DaCe Tiling time1",
                              context={
                                **context,
                                **locals()
                            }) 
        sdfg_list.append(tiled_sdfg)
        time_list.append(time_list[-1] + ttime1[0])
    
    except Exception as e:
        print("DaCe Tiling failed")
        print(e)
        tiled_sdfg = copy.deepcopy(parallel_sdfg)
    '''
                
                
                
    ###########################################################
    ###### Standalone Test Auto - Opt after strict transformation
    try:
        nofuse = False
        if module_name in ['durbin', 'floyd_warshall']:
            #NOTE: These graphs will further fuse (they also validate 
            # and are 50% faster), but the program is not correct/safe.
            # This is a simple access set check I have to implement in SubgraphFusion
            # in the following days, should be easy to fix. 
            nofuse = True
        

        def autoopt(sdfg, device, symbols, nofuse):
            # Mark arrays as on the GPU
            if device == dtypes.DeviceType.GPU:
                for k, v in sdfg.arrays.items():
                    if not v.transient and type(v) == dace.data.Array:
                        v.storage = dace.dtypes.StorageType.GPU_Global

            # Auto-optimize SDFG
            opt.auto_optimize(auto_opt_sdfg, 
                              device, 
                              nofuse = nofuse, 
                              symbols = symbols)

        auto_opt_sdfg = copy.deepcopy(strict_sdfg)
        auto_opt_sdfg._name = 'auto_opt'
        device = dtypes.DeviceType.GPU if finfo["arch"] == "GPU" else dtypes.DeviceType.CPU

        _, auto_time = benchmark(
            f"autoopt(auto_opt_sdfg, device, symbols = locals(), nofuse = nofuse)",
            out_text="DaCe Auto - Opt",
            context={
                **context,
                **locals()
            })
        
        sdfg_list.append(auto_opt_sdfg)
        time_list.append(time_list[-1] + auto_time[0])

    
    except Exception as e:
        print("DaCe autoopt failed")
        print(e)
        traceback.print_exc()
        auto_opt_sdfg = copy.deepcopy(strict_sdfg)
    


    def vectorize(sdfg, vec_len=None):
        matches = []
        for xform in Optimizer(sdfg).get_pattern_matches(
                patterns=[Vectorization]):
            matches.append(xform)
        for xform in matches:
            if vec_len:
                xform.vector_len = vec_len
            xform.apply(sdfg)
    
    # try:
    #     vec_sdfg = copy.deepcopy(parallel_sdfg)
    #     vec_sdfg._name = "vec"
    #     vec_len = None
    #     if finfo["arch"] == "GPU":
    #         vec_len = 2
    #     _, vec_time = benchmark("vectorize(vec_sdfg, vec_len)",
    #                             out_text="DaCe Vectorization time",
    #                             context={**context, **locals()})
    #     sdfg_list.append(vec_sdfg)
    #     time_list.append(time_list[-1] + vec_time[0])
    # except Exception as e:
    #     print("DaCe Vectorization failed")
    #     print(e)

    # sdfg_list = [strict_sdfg, fusion_sdfg]
    # time_list = [parse_time[0] + strict_time[0],
    #              parse_time[0] + strict_time[0] + fusion_time[0]]
    # sdfg_list.append(vec_sdfg)
    # time_list.append(time_list[-1] + vec_time[0])
    if finfo["arch"] == "GPU":
        def_impl = dace.Config.get('library', 'blas', 'default_implementation')
        if def_impl != "pure":
            dace.Config.set('library',
                            'blas',
                            'default_implementation',
                            value='cuBLAS')

    def copy_to_gpu(sdfg):
        for k, v in sdfg.arrays.items():
            if not v.transient and isinstance(v, dace.data.Array):
                v.storage = dace.dtypes.StorageType.GPU_Global

        # Set library nodes
        for node,state in sdfg.all_nodes_recursive(): 
            if isinstance(node, dace.nodes.LibraryNode):
                if node.default_implementation == 'specialize':
                    print("Expanding Node (Common)", node)
                    node.expand(sdfg, state) 

        for node, state in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.LibraryNode):
                from dace.sdfg.scope import is_devicelevel_gpu
                # Use CUB for device-level reductions
                if ('CUDA (device)' in node.implementations 
                     and not is_devicelevel_gpu(state.parent, state, node) 
                     and state.scope_dict()[node] is None):
                    node.implementation = 'CUDA (device)'
                if 'cuBLAS' in node.implementations and not is_devicelevel_gpu(state.parent, state, node):
                    node.implementation = 'cuBLAS'
        
        
    if finfo["arch"] == "GPU":
        # from numba import cuda
        import cupy as cp

    # When all other implementations are too slow, use only autoopt
    if module_name == 'mandelbrot2':
        sdfg_list = [sdfg_list[-1]]
        time_list = [time_list[-1]]
        
    for sdfg, t in zip(sdfg_list, time_list):
        fe_time = t
        if sdfg._name != 'auto_opt':
            device = dtypes.DeviceType.GPU if finfo["arch"] == "GPU" else dtypes.DeviceType.CPU
            dace.transformation.auto_optimize.set_fast_implementations(sdfg, device)
        if finfo["arch"] == "GPU":
            if sdfg._name in ['strict', 'parallel', 'fusion']:
                _, gpu_time1 = benchmark("copy_to_gpu(sdfg)",
                                        out_text="DaCe GPU transformation time1",
                                        context={
                                            **context,
                                            **locals()
                                        })

                _, gpu_time2 = benchmark("sdfg.apply_gpu_transformations()",
                                        out_text="DaCe GPU transformation time2",
                                        context={
                                            **context,
                                            **locals()
                                        })
                _, gpu_time3 = benchmark("sdfg.apply_strict_transformations()",
                                        out_text="DaCe GPU transformation time3",
                                        context={
                                            **context,
                                            **locals()
                                        })
                # NOTE: to be fair, allow one additional greedy MapFusion after GPU trafos
                _, gpu_time4 = benchmark("sdfg.apply_transformations_repeated(MapFusion)",
                                         out_text = "DaCe GPU transformation time4",
                                         context = {**context, **locals()})
                fe_time += gpu_time2[0] + gpu_time3[0] + gpu_time4[0]
            else:
                gpu_time1 = [0]
            fe_time += gpu_time1[0] 
        try:
            dc_exec, compile_time = benchmark("sdfg.compile()",
                                              out_text="DaCe compilation time",
                                              context={
                                                  **context,
                                                  **locals()
                                              })
        except Exception as e:
            print("Failed to compile DaCe {a} {s} implementation.".format(
                a=finfo["arch"], s=sdfg._name))
            print(e)
            traceback.print_exc()
            print("Traceback")
            continue

        fe_time += compile_time[0]

        exec_str = "dc_exec({a})".format(a=finfo["arg_str"])
        setup_str = finfo["setup_str"]
        report_str = finfo["report_str"]
        np_out = context["np_out"]

        out_args = None
        if "out_args" in finfo.keys():
            out_args = finfo["out_args"]

        values, valid = run_mode(exec_str, setup_str, report_str, np_out, mode,
                                 validate, repeat, {
                                     **context,
                                     **locals()
                                 }, out_args, finfo["arch"] == "GPU")

        def dace_runtime(t, mode="main", compile_time=0.0):
            if mode == "first":
                return t + compile_time
            else:
                return t

        if values:
            for vl in values:
                benchmark_values.append(
                    dict(benchmark=module_name,
                         kind=kind,
                         domain=domain,
                         dwarf=dwarf,
                         framework=report_str,
                         mode=mode,
                         details=sdfg._name,
                         validated=valid,
                         time=dace_runtime(vl, mode, fe_time)))

    return benchmark_values


def run_pythran(
    module_name: str,  # Module name
    kind: str,  # Kind, e.g., (micro)benchmark or (micro)application
    domain: str,  # Scientific domain, e.g., CFDs
    dwarf: str,  # Berkeley dwarf, e.g., dense linear algebra
    finfo: Dict[str, Any],  # Framework information
    mode: str = "main",  # Benchmark mode
    validate: bool = True,  # Enables validation against NumPy,
    repeat: int = 10,  # Benchmark repetitions
    context: Dict[Any, Any] = None  # Used to pass test-specific variables
):
    benchmark_values = []

    compile_str = (
        "os.system(\"pythran -DUSE_XSIMD -fopenmp -march=native " +
        "-ffast-math {m}.py -o {m}_opt.so\")".format(m=finfo["module_str"]))
    try:
        cwd = os.getcwd()
        os.chdir(finfo["module_path"])
        _, compile_time = benchmark(compile_str,
                                    out_text="Pythran compilation time")
        os.chdir(cwd)
        exec("from {m}_opt import {f} as ct_impl".format(m=finfo["module_str"],
                                                         f=finfo["func_str"]))
        fe_time = compile_time[0]
    except Exception as e:
        print("Failed to load the Pythran implementation.")
        raise (e)

    exec_str = "ct_impl({a})".format(a=finfo["arg_str"])
    setup_str = finfo["setup_str"]
    report_str = finfo["report_str"]
    np_out = context["np_out"]

    out_args = None
    if "out_args" in finfo.keys():
        out_args = finfo["out_args"]

    values, valid = run_mode(exec_str, setup_str, report_str, np_out, mode,
                             validate, repeat, {
                                 **context,
                                 **locals()
                             }, out_args)

    def pythran_runtime(t, mode="main", compile_time=0.0):
        if mode == "first":
            return t + compile_time
        else:
            return t

    if values:
        for vl in values:
            benchmark_values.append(
                dict(benchmark=module_name,
                     kind=kind,
                     domain=domain,
                     dwarf=dwarf,
                     framework=report_str,
                     mode=mode,
                     details="pythran",
                     validated=valid,
                     time=pythran_runtime(vl, mode, fe_time)))

    return benchmark_values


def run_legate(
    framework: str,  # Framework name
    module_name: str,  # Module name
    func_name: str,  # Function name
    finfo: Dict[str, Dict[str, Any]],  # Framework information
    mode: str = "main",  # Benchmark mode
    validate: bool = True,  # Enables validation against NumPy,
    repeat: int = 10,  # Benchmark repetitions
    append: bool = False,  # Append to CSV file
    context: Dict[Any, Any] = None  # Used to pass test-specific variables
):
    if framework != "legate":
        raise ValueError(
            "This method should be used to benchmark Legate only!")
    info = finfo["legate"]

    # Change directory
    cwd = os.getcwd()
    os.chdir(info["module_path"])

    # Validation
    if validate:
        try:
            exec("from {m} import {f} as np_impl".format(
                m=finfo["numpy"]["module_str"], f=finfo["numpy"]["func_str"]))
            np_exec_str = "np_impl({})".format(finfo["numpy"]["arg_str"])
            if "out_args" in finfo["numpy"].keys():
                ldict = {}
                exec(finfo["numpy"]["setup_str"], {
                    **context,
                    **locals()
                }, ldict)
                for arg in finfo["numpy"]["out_args"]:
                    exec("{a} = ldict[\"{a}\"]".format(a=arg))
            np_out = eval(np_exec_str, {**context, **locals()})
            if "out_args" in finfo["numpy"].keys():
                np_out = [ldict[a] for a in finfo["numpy"]["out_args"]]
            if not isinstance(np_out, (list, tuple)):
                np_out = [np_out]
            np.savez(module_name, *np_out)
        except Exception as e:
            print("Failed to load the NumPy implementation. "
                  "Validation is not possible.")
            print(e)
            validate = False

    os.system("legate {m}_legate.py -v True".format(m=module_name))

    # Change back
    os.chdir(cwd)


def run(
    framework: str,  # Framework name
    module_name: str,  # Module name
    func_name: str,  # Function name
    finfo: Dict[str, Dict[str, Any]],  # Framework information
    mode: str = "main",  # Benchmark mode
    validate: bool = True,  # Enables validation against NumPy,
    repeat: int = 10,  # Benchmark repetitions
    append: bool = False,  # Append to CSV file
    context: Dict[Any, Any] = None  # Used to pass test-specific variables
):
    if framework == "legate":
        return run_legate(framework, module_name, func_name, finfo, mode,
                          validate, repeat, append, context)

    # Validation
    if validate and framework != "numpy":
        try:
            exec("from {m} import {f} as np_impl".format(
                m=finfo["numpy"]["module_str"], f=finfo["numpy"]["func_str"]))
            np_exec_str = "np_impl({})".format(finfo["numpy"]["arg_str"])
            if "out_args" in finfo["numpy"].keys():
                ldict = {}
                exec(finfo["numpy"]["setup_str"], {
                    **context,
                    **locals()
                }, ldict)
                for arg in finfo["numpy"]["out_args"]:
                    exec("{a} = ldict[\"{a}\"]".format(a=arg))
            np_out = eval(np_exec_str, {**context, **locals()})
            if "out_args" in finfo["numpy"].keys():
                np_out = [ldict[a] for a in finfo["numpy"]["out_args"]]
        except Exception as e:
            print("Failed to load the NumPy implementation. "
                  "Validation is not possible.")
            print(e)
            validate = False
            np_out = None
    else:
        validate = False
        np_out = None

    # CSV headers
    file_name = "{mod}_{f}_{m}.csv".format(mod=module_name,
                                           f=framework,
                                           m=mode)
    field_names = [
        "benchmark", "kind", "domain", "dwarf", "framework", "mode", "details",
        "validated", "time"
    ]

    # Extra information
    kind = ""
    if "kind" in finfo.keys():
        kind = finfo["kind"]
    domain = ""
    if "domain" in finfo.keys():
        domain = finfo["domain"]
    dwarf = ""
    if "dwarf" in finfo.keys():
        dwarf = finfo["dwarf"]

    # Special cases
    if framework == "numba":
        bvalues = run_numba(module_name, kind, domain, dwarf, finfo["numba"],
                            mode, validate, repeat, {
                                **context, "np_out": np_out
                            })
    elif framework == "pythran":
        bvalues = run_pythran(module_name, kind, domain, dwarf,
                              finfo["pythran"], mode, validate, repeat, {
                                  **context, "np_out": np_out
                              })
    elif framework in ("dace_cpu", "dace_gpu"):
        bvalues = run_dace(module_name, kind, domain, dwarf, finfo[framework],
                           mode, validate, repeat, {
                               **context, "np_out": np_out
                           })

    else:
        try:
            exec("from {m} import {f} as ct_impl".format(
                m=finfo[framework]["module_str"],
                f=finfo[framework]["func_str"]))
        except Exception as e:
            print("Failed to load the {r} {f} implementation.".format(
                r=finfo[framework]["report_str"],
                f=finfo[framework]["func_str"]))
            raise e
        exec_str = "ct_impl({a})".format(a=finfo[framework]["arg_str"])
        setup_str = finfo[framework]["setup_str"]
        report_str = finfo[framework]["report_str"]

        out_args = None
        if "out_args" in finfo[framework].keys():
            out_args = finfo[framework]["out_args"]

        # For CuPy
        if framework == "cupy":
            import cupy as cp

        bvalues = []
        values, valid = run_mode(exec_str, setup_str, report_str, np_out, mode,
                                 validate, repeat, {
                                     **context,
                                     **locals()
                                 }, out_args, framework == "cupy", framework)

        if mode == "main" and values:
            for vl in values:
                bvalues.append(
                    dict(benchmark=module_name,
                         kind=kind,
                         domain=domain,
                         dwarf=dwarf,
                         framework=report_str,
                         mode=mode,
                         details=framework,
                         validated=valid,
                         time=vl))

    write_csv(file_name, field_names, bvalues, append=append)


# def run(exec_str, setup_str, report_str, np_out=None,
#         mode="median", validate=True, repeat=10, append=True, context=None):
#     if mode not in ("first", "median", "papi", "tau"):
#         raise ValueError("Invalid input {} for mode keyword argument.".format(mode))

#     # Always run "first"
#     try:
#         ct_out, fe_time = benchmark(exec_str, setup=setup_str, out_text=report_str + " first execution", context=context)
#     except Exception as e:
#         print("Failed to benchmark {} first execution.".format(report_str))
#         print(e)
#         return None, None

#     if mode == "median":
#         valid = None
#         if validate and np_out is not None:
#             try:
#                 valid = validation(np_out, ct_out, report_str)
#             except Exception as e:
#                 print("Failed to run {} validation.".format(report_str))
#         try:
#             _, me_time = benchmark(exec_str, setup=setup_str, out_text=report_str + " median execution", repeat=repeat, context=context)
#         except Exception as e:
#             print("Failed to benchmark {} median execution.".format(report_str))
#             return None, None

#         return me_time, valid

#     if mode == "papi":
#         try:
#             from pypapi import papi_high
#             from pypapi import events as papi_events

#             events = [papi_events.PAPI_DP_OPS]
#             # events = [papi_events.PAPI_L2_TCM, papi_events.PAPI_L2_TCA]
#             # events = [papi_events.PAPI_L3_TCM, papi_events.PAPI_L3_TCA]

#             if setup_str != "pass":
#                 eval(setup_str, context)
#             papi_high.start_counters(events)
#             eval(exec_str, globals())
#             counters = papi_high.read_counters()
#             papi_high.stop_counters()
#         except Exception as e:
#             print("Failed to benchmark {} PAPI counters .".format(report_str))
#             raise e

#         return None, None

#     if mode == "tau":
#         try:
#             import tau
#             # import pytau

#             if setup_str != "pass":
#                 eval(setup_str, context)

#             # x = pytau.profileTimer("TAU profiling")
#             # pytau.start(x)
#             # eval(exec_str, context)
#             # pytau.stop(x)
#             # pytau.dbDump()
#             tau.run(exec_str)
#         except Exception as e:
#             print("Failed to benchmark {} PAPI counters .".format(report_str))
#             raise e

#         return None, None

# def run_numba(module_name, arg_str, setup_str, np_out=None, mode="median",
#               validate=True, repeat=10, append=True, context=None):

#     file_name = "{mod}_numba_{m}.csv".format(mod=module_name, m=mode)
#     field_names = ["benchmark", "framework", "mode", "details", "validated", "time"]
#     values = []

#     # for func_name in ("object_mode", "object_mode_prange",
#     #                   "nopython_mode", "nopython_mode_prange"):
#     for func_name in ("object_mode", "object_mode_parallel", "object_mode_prange",
#                       "nopython_mode", "nopython_mode_parallel", "nopython_mode_prange"):
#         # Import Numba implementation
#         try:
#             exec("from {m}_numba import {f} as ct_impl".format(
#                 m=module_name, f=func_name))
#         except Exception as e:
#             print("Failed to load the Numba implementation.")
#             print(e)
#             continue

#         # for parallel, fastmath in itertools.product([False, True],
#         #                                             [False, True]):
#         #     post_str = "parallel={p}, fastmath={f}".format(p=parallel,
#         #                                                    f=fastmath)
#         #     exec_str = "ct_impl({a}, {p})".format(a=arg_str, p=post_str)
#         #     report_str = "Numba {f} ({p})".format(f=func_name, p=post_str)
#         exec_str = "ct_impl({a})".format(a=arg_str)
#         report_str = "Numba {f}".format(f=func_name)

#         nval, valid = run(exec_str, setup_str, report_str, np_out,
#                             mode, validate, repeat, append, {**context, **locals()})

#         if mode == "median" and nval:
#             for v in nval:
#                 values.append(dict(
#                     benchmark=module_name,
#                     framework="Numba",
#                     mode=mode,
#                     # details="{f} ({p})".format(f=func_name, p=post_str),
#                     details=func_name,
#                     validated=valid,
#                     time=v
#                 ))

#     write_csv(file_name, field_names, values)

# def run_dace(module_name, func_name, arg_str, setup_str, np_out=None, arch="cpu",
#              mode="median", validate=True, repeat=10, append=True, context=None):

#     file_name = "{mod}_dace_{a}_{m}.csv".format(mod=module_name, a=arch, m=mode)
#     field_names = ["benchmark", "framework", "mode", "details", "validated", "time"]
#     values = []

#     # Import DaCe implementation
#     try:
#         from dace.transformation.optimizer import Optimizer
#         from dace.transformation.dataflow import MapFusion, Vectorization
#         exec("from {m}_dace import {f} as ct_impl".format(
#             m=module_name, f=func_name))
#     except Exception as e:
#         print("Failed to load the DaCe implementation.")
#         raise(e)

#     dc_sdfg, _ = benchmark("ct_impl.to_sdfg(strict=False)",
#                            out_text="DaCe parsing time",
#                            context={**context, **locals()})
#     _, _ = benchmark("dc_sdfg.apply_strict_transformations()",
#                      out_text="DaCe Strict Transformations time",
#                      context={**context, **locals()})
#     _, _ = benchmark("dc_sdfg.apply_transformations_repeated([MapFusion])",
#                      out_text="DaCe MapFusion time",
#                      context={**context, **locals()})

#     def vectorize(sdfg):
#         matches = []
#         for xform in Optimizer(sdfg).get_pattern_matches(
#                 patterns=[Vectorization]):
#             matches.append(xform)
#         for xform in matches:
#             # xform.vector_len = 2
#             xform.apply(sdfg)

#     # _, _ = benchmark("vectorize(dc_sdfg)",
#     #                  out_text="DaCe Vectorization time",
#     #                  context={**context, **locals()})
#     # dc_exec, _ = benchmark("ct_impl.compile()",
#     dc_exec, _ = benchmark("dc_sdfg.compile()",
#                            out_text="DaCe compilation time",
#                            context={**context, **locals()})

#     exec_str = "dc_exec({a})".format(a=arg_str)
#     report_str = "DaCe {a}".format(a=arch)

#     nval, valid = run(exec_str, setup_str, report_str, np_out,
#                         mode, validate, repeat, append, {**context, **locals()})

#     if mode == "median" and nval:
#         for v in nval:
#             values.append(dict(
#                 benchmark=module_name,
#                 framework=report_str,
#                 mode=mode,
#                 details=None,
#                 validated=valid,
#                 time=v
#             ))

#     write_csv(file_name, field_names, values)
