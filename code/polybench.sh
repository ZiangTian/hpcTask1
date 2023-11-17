#!/bin/bash

# benchmarks=("adi" "atax" "bicg" "cholesky" "correlation" "covariance" "deriche"
#             "doitgen" "durbin" "fdtd_2d")
# benchmarks=("floyd_warshall" "gemm" "gemver" "gesummv" "gramschmidt" "heat_3d"
#             "jacobi_1d" "jacobi_2d" "k2mm" "k3mm")
#benchmarks=("adi" "atax" "bicg" "cholesky" "correlation" "covariance" "deriche"
#            "doitgen" "durbin" "fdtd_2d" "floyd_warshall" "gemm" "gemver"
#            "gesummv" "gramschmidt" "heat_3d" "jacobi_1d" "jacobi_2d" "k2mm"
#            "k3mm" "lu" "ludcmp" "mvt" "nussinov" "seidel_2d" "symm" "syr2k"
#            "syrk" "trisolv" "trmm")
benchmarks=("floyd_warshall" "gemm" "gemver"
            "doitgen" "durbin" "fdtd_2d" "floyd_warshall" "gemm" "gemver"
            "gesummv" "gramschmidt" "heat_3d" "jacobi_1d" "jacobi_2d" "k2mm"
            "k3mm" "lu" "ludcmp" "mvt" "nussinov" "seidel_2d" "symm" "syr2k"
            "syrk" "trisolv" "trmm")
#benchmarks=("lu" "ludcmp" "mvt" "nussinov" "seidel_2d" "symm" "syr2k"
#            "syrk" "trisolv" "trmm")
#frameworks=("numpy" "numba" "pythran" "dace_cpu")
frameworks=("numba")

for i in "${benchmarks[@]}"
do
    echo $i
    for j in "${frameworks[@]}"
    do
	    timeout 1000s python npbench/benchmarks/polybench/$i/$i.py -f $j
    done
done
