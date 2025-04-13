#!/bin/bash

num_workers=25  # Specify the number of worker cores here
num_ddh_requests=2500
num_dmdp_requests=0
num_dg_requests=0
num_mg_requests=0
num_ufh_requests=0
num_mmp_requests=0
num_requests=$((num_ddh_requests + num_dmdp_requests + num_dg_requests + num_mg_requests + num_ufh_requests + num_mmp_requests))
num_coroutines=128
load=0.01
exe=main_xrt
dist=3
peak=$((  1 * 1024 ))
peak2=$(( 1024 * 1024 ))
mode=Yield

# make ${exe}_clean
make ${exe} -j
allowd_dev=0.1
echo "Running with load: $load"

worker_types=( 3 4 1 5 2 )
declare -A results
declare -A avg_slowdown
declare -A median_slowdown

for worker_type in "${worker_types[@]}"; do
  echo "Running sudo stdbuf -o0 ./${exe} -n ${num_requests} -z ${num_ddh_requests} -x ${num_dmdp_requests} -h ${num_dg_requests} -g ${num_mg_requests} -f ${num_ufh_requests} -a ${num_mmp_requests} -c ${num_coroutines} -w ${num_workers} -i 2 -p ${load} -s 26 -l 5 -v ${dist} -b ${peak} -k ${peak2} -m ${worker_type} -j 1 -l 5"

  sudo stdbuf -o0 ./${exe} -n ${num_requests} \
    -z ${num_ddh_requests} -x ${num_dmdp_requests} \
    -h ${num_dg_requests} -g ${num_mg_requests} \
    -f ${num_ufh_requests} -a ${num_mmp_requests} \
    -c ${num_coroutines} -w ${num_workers} -i 2 -p ${load} \
    -s 26 -l 5 -v ${dist} \
    -b ${peak} -k ${peak2} -m ${worker_type} -j 1 \
    -l 5 \
    | tee ${exe}-${num_ddh_requests}DDH-${num_dmdp_requests}DMDP-${num_dg_requests}DG-${num_mg_requests}MG-${num_ufh_requests}UFH-${num_mmp_requests}MMP-${mode}-${num_workers}w-${num_coroutines}c-${dist}dist-${peak}-${peak2}bimodal-${worker_type}type-$load.txt

  echo "Load MedianLatency 99thPercentileLatency CompletionRate"
  i=${exe}-${num_ddh_requests}DDH-${num_dmdp_requests}DMDP-${num_dg_requests}DG-${num_mg_requests}MG-${num_ufh_requests}UFH-${num_mmp_requests}MMP-${mode}-${num_workers}w-${num_coroutines}c-${dist}dist-${peak}-${peak2}bimodal-${worker_type}type-${load}.txt
  load=$(echo $i | awk -F'-' '{print $15}' | sed 's/.txt//')
  echo "processing $i"
  echo -n "$load "
  results[$worker_type]=$(grep -E "99.9th Percentile Slowdown" $i | awk '{print $NF}')
  avg_slowdown[$worker_type]=$(grep -E "Average Slowdown" $i | awk '{print $NF}')
  median_slowdown[$worker_type]=$(grep -E "Median Slowdown" $i | awk '{print $NF}')
done

echo -e "Load\tNoAcc\tBlock&Wait\tRR\tRR-SW-Fallback\tMS-SW-Fallback"
echo -e "99.9pSlowdown\t$load\t${results[3]}\t${results[4]}\t${results[1]}\t${results[5]}\t${results[2]}"
echo -e "AverageSlowdown\t$load\t${avg_slowdown[3]}\t${avg_slowdown[4]}\t${avg_slowdown[1]}\t${avg_slowdown[5]}\t${avg_slowdown[2]}"
echo -e "MedianSlowdown\t$load\t${median_slowdown[3]}\t${median_slowdown[4]}\t${median_slowdown[1]}\t${median_slowdown[5]}\t${median_slowdown[2]}"
