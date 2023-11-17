#!/bin/bash

# benchmarks=("adi" "atax" "bicg" "cholesky" "correlation" "covariance" "deriche"
#             "doitgen" "durbin" "fdtd_2d")
# benchmarks=("floyd_warshall" "gemm" "gemver" "gesummv" "gramschmidt" "heat_3d"
#             "jacobi_1d" "jacobi_2d" "k2mm" "k3mm")
benchmarks=("adi" "atax" "bicg" "cholesky" "correlation" "covariance" "deriche"
            "doitgen" "durbin" "fdtd_2d" "floyd_warshall" "gemm" "gemver"
            "gesummv" "gramschmidt" "heat_3d" "jacobi_1d" "jacobi_2d" "k2mm"
            "k3mm" "lu" "ludcmp" "mvt" "nussinov" "seidel_2d" "symm" "syr2k"
            "syrk" "trisolv" "trmm")
#frameworks=("cupy" "dace_gpu")
frameworks=("dace_gpu")

for i in "${benchmarks[@]}"
do
    for j in "${frameworks[@]}"
    do
	    timeout 500s python npbench/benchmarks/polybench/$i/$i.py -f $j
    done
done
