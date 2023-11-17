#!/bin/bash
#SBATCH -t 04:00:00

#frameworks=("numpy" "numba" "pythran" "dace_cpu" "cupy" "dace_gpu")
frameworks=("numba")

benchmarks=("azimint_hist" "azimint_naive"
            "cavity_flow" "channel_flow"
            "compute"
            "contour_integral"
            "crc16"
            "go_fast"
            "mandelbrot1" "mandelbrot2"
            "nbody"
            "spmv"
            "scattering_self_energies"
            "stockham_fft")

deep_learning=("conv2d_bias" "softmax" "mlp" "lenet" "resnet")

weather=("hdiff" "vadv")

polybench=("adi" "atax" "bicg" "correlation" "covariance" "deriche" "doitgen" "durbin" "fdtd_2d"
           "floyd_warshall" "gemm" "gemver" "gesummv" "gramschmidt" "heat_3d" "jacobi_1d"
           "jacobi_2d" "k2mm" "k3mm" "lu" "ludcmp" "mvt" "nussinov" "seidel_2d" "symm"
           "syr2k" "syrk" "trisolv" "trmm")

pythran=("arc_distance")


echo $PYTHONPATH
for i in "${benchmarks[@]}"
do
    echo ---------------------------------------------------------------------------
    echo $i
    for j in "${frameworks[@]}"
    do
	    timeout 500s python npbench/benchmarks/$i/$i.py -f $j
    done
done

for i in "${deep_learning[@]}"
do
    echo ---------------------------------------------------------------------------
    echo $i
    for j in "${frameworks[@]}"
    do
	    timeout 500s python npbench/benchmarks/deep_learning/$i/$i.py -f $j
    done
done

for i in "${weather[@]}"
do
    echo ----------------------------------------------------------------------------
    echo $i
    for j in "${frameworks[@]}"
    do
        timeout 500s python npbench/benchmarks/weather_stencils/$i/$i.py -f $j
    done
done

#for i in "${polybench[@]}"
#do
#    echo ---------------------------------------------------------------------------
#    echo $i
#    for j in "${frameworks[@]}"
#    do
#        timeout 500s python npbench/benchmarks/polybench/$i/$i.py -f $j
#    done
#done

#for i in "${pythran[@]}"
#do
#    echo $i
#    for j in "${frameworks[@]}"
#    do
#        timeout 500s python npbench/benchmarks/pythran/$i/$i.py -f $j
#    done
#done


