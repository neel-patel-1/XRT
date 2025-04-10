#!/bin/bash

EXES=( worker_vs_dispatcher worker_notified_forward )
MODES=( Yield )
num_workers=25  # Specify the number of worker cores here
num_requests=26000
LOAD_LEVELS=( 0.3 0.5 1 1.25 1.5 2 2.3 2.5 2.75 3 3.5 )

mode=Yield
for exe in ${EXES[@]}; do
  #rm -f ${exe}
  make ${exe} CXXFLAGS="-g -O3 -DLATENCY -DTHROUGHPUT -DPOISSON -DPERF -DJSQ -D${mode} "  -j

  prefix=logs/${exe}-DDH${mode}-${num_workers}w-4c
  allowd_dev=0.1
  for load in "${LOAD_LEVELS[@]}"; do
    echo "Running with load: $load"
    sudo stdbuf -o0 ./${exe} -n ${num_requests} -c 4 -w ${num_workers} -i 2 -p ${load} -s 26 -l 5 \
      2>/dev/null > ${prefix}-$load.txt
  done

  # Find an unused filename for the summary
  ctr=0
  while [[ -e logs/${exe}_summary.txt.${ctr} ]]; do
    ((ctr++))
  done
  summary_file=logs/${exe}_summary.txt.${ctr}

  #sorted by the load
  echo "Load AverageLatency 99thPercentileLatency CompletionRate" > ${summary_file}
  for i in $( ls -1 ${prefix}* | sort -V -k5 -t- ); do
    load=$(echo $i | awk -F'-' '{print $5}' | sed 's/.txt//')
    echo -n "$load " >> ${summary_file}
    grep -E "Response Rate|Average Latency|99th Percentile Latency" $i | awk '{printf "%s ", $NF}' | \
      awk '{print $3 " " $2 " " $1}' >> \
      ${summary_file}
  done
done