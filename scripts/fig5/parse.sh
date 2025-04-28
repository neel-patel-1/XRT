#!/bin/bash

PAYLOAD_SIZES=( 256 1024 4096 16384 65536 262144 1048576  )

bin=throughput
APPS=( 0 1 2 7 10 11 )
for app in "${APPS[@]}"; do
  prefix="logs/$(basename $bin)"_${app}
  for i in `seq 8 2 20`; do
    grep -e 'Block&Wait' ${prefix}_${i}.log | head -n 1 \
    | ./scripts/fig3/transpose.sh \
    | grep -v -e 'Block&Wait' | grep -v $(( 2 ** i )) | tee logs/results_${bin}_${app}.log ;
    echo "" | tee -a logs/results_${bin}_${app}.log ;
    grep NoAcceleration  ${prefix}_${i}.log | head -n 1 \
    | ./scripts/fig3/transpose.sh \
    | awk '{if(NR!=4) {print}}' | grep -v NoAcceleration | grep -v $(( 2 ** i )) | tee -a logs/results_${bin}_${app}.log;
  done
done

 #for i in `seq 8 2 20`; do grep Blocking $(( 2 ** i ))_exe_time_mg.log | head -n 1 |  scripts/transpose.sh | grep -v Blocking | grep -v $(( 2 ** i )) | tee mg_$(( 2 ** i )).log ; echo "" | tee -a mg_$(( 2 ** $i )).log ;  grep Baseline  $(( 2 ** i ))_exe_time_mg.log | head -n 1 | scripts/transpose.sh | awk '{if(NR!=4) {print}}' | grep -v Baseline | grep -v $(( 2 ** i )) | tee -a mg_$((2 **i)).log;  done