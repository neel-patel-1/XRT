#!/bin/bash

num_workers=25  # Specify the number of worker cores here
num_ddh_requests=0
num_dmdp_requests=0
num_dg_requests=0
num_mg_requests=26000
num_ufh_requests=0
num_mmp_requests=0
num_requests=$((num_ddh_requests + num_dmdp_requests + num_dg_requests + num_mg_requests + num_ufh_requests + num_mmp_requests))
num_coroutines=128
exe=blocking_overhead
dist=3
peak=$((  16 * 1024 ))
peak2=$(( 1024 * 1024 ))
mode=Yield

LOADS=( 0.5 )

# make ${exe}_clean
make ${exe} CXXFLAGS="-g -O3 -DLATENCY -DTHROUGHPUT -DPERF -DPOISSON -DJSQ -DLOST_ENQ_TIME"  -j
allowd_dev=0.1

worker_types=( 3 4 1 )
declare -A results
declare -A avg_request_time
declare -A ninety_ninth_percentile
declare -A avg_slowdown
declare -A ninety_ninth_percentile_slowdown
declare -A enq_rate
declare -A response_rate

declare -A all_loads

prefix=logs/${exe}-${num_ddh_requests}DDH-${num_dmdp_requests}DMDP-${num_dg_requests}DG-${num_mg_requests}MG-${num_ufh_requests}UFH-${num_mmp_requests}MMP-${mode}-${num_workers}w-${num_coroutines}c-${dist}dist-${peak}-${peak2}bimodal

load=0.5
while true; do
  all_loads[$load]=1
  for worker_type in "${worker_types[@]}"; do
    echo "Running worker ${worker_type} with load: $load"
    output="${prefix}-${worker_type}type-$load.txt"
    sudo stdbuf -o0 ./${exe} -n ${num_requests} \
      -z ${num_ddh_requests} -x ${num_dmdp_requests} \
      -h ${num_dg_requests} -g ${num_mg_requests} \
      -f ${num_ufh_requests} -a ${num_mmp_requests} \
      -c ${num_coroutines} -w ${num_workers} -i 2 -p ${load} \
      -s 26 -l 5 -v ${dist} \
      -b ${peak} -k ${peak2} -m ${worker_type} -j 1 \
      -l 5 \
      2>/dev/null > ${output}
    results[$load,$worker_type]=$(grep -E "Average Lost Enqueue Time" ${output} | awk '{print $NF}')
    avg_request_time[$load,$worker_type]=$(grep -E "Average Time per Request" ${output} | awk '{print $NF}')
    ninety_ninth_percentile[$load,$worker_type]=$(grep -E "99th Percentile Lost Enqueue Time" ${output} | awk '{print $NF}')
    avg_slowdown[$load,$worker_type]=$(grep -E "Average Slowdown" ${output} | awk '{print $NF}')
    ninety_ninth_percentile_slowdown[$load,$worker_type]=$(grep -E "99.9th Percentile Slowdown" ${output} | awk '{print $NF}')
    enq_rate[$load,$worker_type]=$(grep -E "Enqueue Rate" ${output} | awk '{print $NF}')
    response_rate[$load,$worker_type]=$(grep -E "Response Rate" ${output} | awk '{print $NF}')
  done
  if (( $(echo " $load > ${enq_rate[$load,4]} + 1.0  " | bc -l) )); then
    break
  fi
  load=$(echo "$load + 0.5" | bc)
done

ctr=0
while [[ -e logs/blocking_summary.txt.${ctr} ]]; do
  ((ctr++))
done
summary_file=logs/blocking_summary.txt.${ctr}

echo -e "Name\tLoad\tNoAcc\tBlock&Wait\tRR" > ${summary_file}

for load in $(echo "${!all_loads[@]}" | tr ' ' '\n' | sort -n); do
  echo -e "AvgRequestTime\t$load\t${avg_request_time[$load,3]}\t${avg_request_time[$load,4]}\t${avg_request_time[$load,1]}" >> ${summary_file}
done

for load in $(echo "${!all_loads[@]}" | tr ' ' '\n' | sort -n); do
  echo -e "AvgSlowdown\t$load\t${avg_slowdown[$load,3]}\t${avg_slowdown[$load,4]}\t${avg_slowdown[$load,1]}" >> ${summary_file}
done

for load in $(echo "${!all_loads[@]}" | tr ' ' '\n' | sort -n); do
  echo -e "99pSlowdown\t$load\t${ninety_ninth_percentile_slowdown[$load,3]}\t${ninety_ninth_percentile_slowdown[$load,4]}\t${ninety_ninth_percentile_slowdown[$load,1]}" >> ${summary_file}
done

for load in $(echo "${!all_loads[@]}" | tr ' ' '\n' | sort -n); do
  echo -e "AvgEnqueueTime\t$load\t${results[$load,3]}\t${results[$load,4]}\t${results[$load,1]}" >> ${summary_file}
done

for load in $(echo "${!all_loads[@]}" | tr ' ' '\n' | sort -n); do
  echo -e "99.9pEnqueueTime\t$load\t${ninety_ninth_percentile[$load,3]}\t${ninety_ninth_percentile[$load,4]}\t${ninety_ninth_percentile[$load,1]}" >> ${summary_file}
done

for load in $(echo "${!all_loads[@]}" | tr ' ' '\n' | sort -n); do
  echo -e "EnqueueRate\t$load\t${enq_rate[$load,3]}\t${enq_rate[$load,4]}\t${enq_rate[$load,1]}" >> ${summary_file}
done

for load in $(echo "${!all_loads[@]}" | tr ' ' '\n' | sort -n); do
  echo -e "ResponseRate\t$load\t${response_rate[$load,3]}\t${response_rate[$load,4]}\t${response_rate[$load,1]}" >> ${summary_file}
done
