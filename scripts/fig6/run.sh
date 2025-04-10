#!/bin/bash

num_workers=25  # Specify the number of worker cores here
num_ddh_requests=${1:-0}
num_dmdp_requests=${2:-2600}
num_dg_requests=${3:-0}
num_mg_requests=${4:-0}
num_ufh_requests=${5:-0}
num_mmp_requests=${6:-0}
num_requests=$((num_ddh_requests + num_dmdp_requests + num_dg_requests + num_mg_requests + num_ufh_requests + num_mmp_requests))
num_coroutines=128
exe=main_xrt
dist=0
peak=$((  1024 ))
peak2=$(( 1024 * 1024 ))
mode=Yield

make ${exe} CXXFLAGS="-g -O3 -DLATENCY -DTHROUGHPUT -DPERF -DPOISSON -DJSQ -DSW_FALLBACK"  -j

allowd_dev=0.1
worker_types=( 3 4 1 2 )

echo -e "Load\tWorker3\tWorker4\tWorker1\tWorker2"

load=0.3
declare -A all_results
last_non_violating_load=0

while true; do
  declare -A results
  all_above_200=true

  for worker_type in "${worker_types[@]}"; do
    echo "Running worker ${worker_type} with load: $load"

    prefix=logs/${exe}-${num_ddh_requests}DDH-${num_dmdp_requests}DMDP-${num_dg_requests}DG-${num_mg_requests}MG-${num_ufh_requests}UFH-${num_mmp_requests}MMP-${mode}-${num_workers}w-${num_coroutines}c-${dist}dist-${peak}-${peak2}bimodal-${worker_type}type

    sudo stdbuf -o0 ./${exe} -n ${num_requests} \
      -z ${num_ddh_requests} -x ${num_dmdp_requests} \
      -h ${num_dg_requests} -g ${num_mg_requests} -f ${num_ufh_requests} -a ${num_mmp_requests} \
      -c ${num_coroutines} -w ${num_workers} -i 2 -p ${load} -s 26 -l 5 -v ${dist} \
      -b ${peak} -k ${peak2} -m ${worker_type} -j 1 \
      2>/dev/null > ${prefix}-$load.txt

    #sorted by the load
    filename="${prefix}-$load.txt"
    load_value=$(echo $filename | awk -F'-' '{print $13}' | sed 's/.txt//')
    results[$worker_type]=$(grep -E "99.9th Percentile Slowdown" $filename | awk '{print $NF}')

    if (( $(echo "${results[$worker_type]} < 200" | bc -l) )); then
      all_above_200=false
    fi
  done

  all_results[$load]="${results[3]}\t${results[4]}\t${results[1]}\t${results[2]}"

  if $all_above_200; then
    echo "All above 200"
    break
  fi

  last_non_violating_load=$load
  load=$(echo "$load * 2" | bc -l)
done

# Collect 4 additional points between the last non-violating load and the violating one
step=$(echo "($load - $last_non_violating_load) / 5" | bc -l)
for i in {1..4}; do
  load=$(echo "$last_non_violating_load + $step * $i" | bc -l)
  declare -A results

  for worker_type in "${worker_types[@]}"; do
    echo "Running with load: $load and worker type: $worker_type"
    prefix=logs/${exe}-${num_ddh_requests}DDH-${num_dmdp_requests}DMDP-${num_dg_requests}DG-${num_mg_requests}MG-${num_ufh_requests}UFH-${num_mmp_requests}MMP-${mode}-${num_workers}w-${num_coroutines}c-${dist}dist-${peak}-${peak2}bimodal-${worker_type}type

    sudo stdbuf -o0 ./${exe} -n ${num_requests} \
      -z ${num_ddh_requests} -x ${num_dmdp_requests} \
      -h ${num_dg_requests} -g ${num_mg_requests} \
      -c ${num_coroutines} -w ${num_workers} -i 2 -p ${load} -s 26 -l 5 -v ${dist} \
      -b ${peak} -k ${peak2} -m ${worker_type} -j 1 \
      2>/dev/null > ${prefix}-$load.txt

    #sorted by the load
    filename="${prefix}-${load}.txt"
    load_value=$(echo $filename | awk -F'-' '{print $13}' | sed 's/.txt//')
    echo "processing $filename"
    echo -n "$load_value "
    results[$worker_type]=$(grep -E "99.9th Percentile Slowdown" $filename | awk '{print $NF}')
  done

  all_results[$load]="${results[3]}\t${results[4]}\t${results[1]}\t${results[2]}"
  echo -e "$load\t${results[3]}\t${results[4]}\t${results[1]}\t${results[2]}"
done

# find unused filename for the summary
ctr=0
while [[ -e ${filename}.${ctr} ]]; do
  ((ctr++))
done
filename="${prefix}-all.txt.${ctr}"

echo "Load AverageLatency 99thPercentileLatency CompletionRate" > ${filename}
for load in "${!all_results[@]}"; do
  echo -e "$load\t${all_results[$load]}"
done | sort -n >> ${filename}

echo -e "\nAll Load Levels Collected:"