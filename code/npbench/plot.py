import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import glob
import pandas


CMD_SUMMARY = False
PRODUCE_PLOT = True
#PATHLIST = "benchmarks/*/*.csv"
PATHLIST = "../plotting/results_03_16/*.csv"
INCLUDE_VALIDATED = False 




def bootstrap_ci(
    data, 
    statfunction=np.median, 
    alpha = 0.05, 
    n_samples = 300):

    """inspired by https://github.com/cgevans/scikits-bootstrap"""
    import warnings

    def bootstrap_ids(data, n_samples=100):
        for _ in range(n_samples):
            yield np.random.randint(data.shape[0], size=(data.shape[0],))    
    
    alphas = np.array([alpha/2, 1 - alpha/2])
    nvals = np.round((n_samples - 1) * alphas).astype(int)
    if np.any(nvals < 10) or np.any(nvals >= n_samples-10):
        warnings.warn("Some values used extremal samples; results are probably unstable. "
                      "Try to increase n_samples")

    data = np.array(data)
    if np.prod(data.shape) != max(data.shape):
        raise ValueError("Data must be 1D")
    data = data.ravel()
    
    boot_indexes = bootstrap_ids(data, n_samples)
    stat = np.asarray([statfunction(data[_ids]) for _ids in boot_indexes])
    stat.sort(axis=0)

    return stat[nvals][1] - stat[nvals][0]




pathlist = glob.glob(PATHLIST)
data = None
for f in pathlist:
    print("Reading CSV file", f)
    ndata = pandas.read_csv(f)
    if data is None:
        data = ndata
    else:
        data = pandas.concat([data, ndata])

data = data.drop(['kind', 'dwarf'], axis = 1).reset_index(drop = True)

data = data[data['validated'] == True]
data = data.drop(['validated'], axis=1).reset_index(drop = True)


aggdata = data.groupby(["benchmark", "domain", "framework", "mode", "details"], dropna=False).agg({"time": [np.median, bootstrap_ci]}).reset_index()
aggdata = aggdata.drop(["domain", "framework"], axis = 1).reset_index(drop = True)

# benchmark, mode,      details, time,  CI
# gemm       DaCe CPU   fusion   0.3s  

benchmarks = list(sorted(set(aggdata["benchmark"])))
benchmarks_split = [benchmarks[i:i+1] for i in range(0,len(benchmarks), 1)]

if CMD_SUMMARY:
    np.set_printoptions(threshold=sys.maxsize)
    print(aggdata.to_numpy())

if PRODUCE_PLOT:
    print("PLOT")
    M = len(set(aggdata["benchmark"]))
    N = len(set(aggdata["details"]))

    width = 0.65

    subplot_width = 5
    subplot_depth = M // subplot_width
    if M % subplot_width > 0:
        subplot_depth += 1
    fig, axes = plt.subplots(figsize = (25, 25), nrows = subplot_depth, ncols = subplot_width)
    plt.subplots_adjust(hspace=0.25, wspace=0.3)

    #fig.tight_layout()
    #fig, axes = plt.subplots()

    def yield_axes():
        for i in range(M):
            yield (i // subplot_width, i % subplot_width)

    #colors = ['green', 'blue', 'purple', 'red', 'pink']
    colors = ['blue', 'pink', 'purple', 'red', 'green']
    #numpy, strict, fusion, parallel, auto_opt
    #x = [x[2], x[4], x[1], x[3], x[0]]
    def l_sort(tup):
        index = tup[0]
        permutation = [4, 2, 0, 3, 1]
        return permutation[index]


    axes_iter = iter(yield_axes())
    for i in range(M):
        benchmark = benchmarks[i]
        current_axis = next(axes_iter)
        print(benchmark, "->", current_axis)
        
        x_outer = []
        std_outer = []

        for z,(d, (details, detail_group)) in enumerate(sorted(enumerate(aggdata.groupby("details")), key = l_sort)):
            
            detail_group = detail_group.reset_index()
            index_to_drop = [j for j in range(len(detail_group)) if detail_group["benchmark"][j] not in benchmarks_split[i]]
            detail_group = detail_group.drop(index_to_drop)
            local_array = detail_group.to_numpy()
            # benchmark, mode, details, median, bootstrap_ci
            #     0       1       2        3         4

            local_benchmarks = benchmarks_split[i]
            current_benchmarks = [a[1] for a in local_array]
            #print("local benchmarks=", local_benchmarks)
            #print("current benchmarks=", current_benchmarks)

            # insert missing values 
            for bench in local_benchmarks:
                if bench not in current_benchmarks:
                    array_to_append = np.ndarray((1,6), dtype = object)
                    array_to_append[0,:] = [0, bench, 'main', details, 0, 0]
                    local_array = np.append(local_array, array_to_append, axis=0)
                    print("Missing!", bench)
            
            #local_array.sort(axis = 0)
            local_array = local_array[local_array[:,1].argsort()] 

            
            x = [l[4] for l in local_array]
            std = [l[5] for l in local_array]

            x_outer.append(x[0])
            std_outer.append(std[0])  
          
            '''
            rect = axes[current_axis].bar(r, x, 
                    color = colors[z],
                    width = width,
                    edgecolor = 'white',
                    label = details,
                    yerr = std)
            '''
        
        r = np.arange(5)
        try:
            min_h = min(v for v in x_outer if v > 0)
        except ValueError:
            min_h = 0.1
        max_h = 10*min_h

        x_outer_norm = []
        for v in x_outer:
            if v < max_h:
                x_outer_norm.append(v)
            else:
                x_outer_norm.append(max_h)

        rects = axes[current_axis].bar(r, x_outer_norm, color = colors, edgecolor = 'white', yerr = std_outer, alpha = 0.5)
        axes[current_axis].set_xticks([])
        for rect, v in zip(rects, x_outer):
            if v > max_h:
                axes[current_axis].annotate(
                    '{:.1f}x'.format(v / max_h),
                    xy = (rect.get_x() + rect.get_width() / 2, max_h),
                    xytext = (0, -15),
                    textcoords = 'offset points',
                    ha = 'center',
                    va = 'bottom')
            
        
        axes[current_axis].title.set_text(benchmark)
        last_axis = axes[current_axis]
        #plt.title("Runtime Chart")
        #plt.legend()
        '''   
        plt.xlabel('Mode')
        plt.ylabel('Runtime')
        '''
        #plt.xticks([r + ((N-1)*width / 2) for r in range(1)], benchmarks_split[i])
        '''
        plt.legend()
        plt.show()
        '''


    
    st = plt.suptitle('NumPy Benchmark Suite', fontsize = 32)
    st.set_y(0.975)
    fig.subplots_adjust(top=0.925)   
    #fig.subplots_adjust(right=0.9)
    
    handles, labels = last_axis.get_legend_handles_labels()
    fig.legend(handles, labels,prop={"size":18})

    fig.text(0.05, 0.5, 'Runtime (seconds)', va='center', rotation='vertical', fontsize=14)


    #plt.ylabel('Runtime (s)')


    plt.show()
    

