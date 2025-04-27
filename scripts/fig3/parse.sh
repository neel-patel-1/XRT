#!/bin/bash

bin=exe_time

APPS=( 0 1 2 7 10 11 )
sizes=( `seq 8 2 20` );

for app in "${APPS[@]}"; do
  prefix="logs/$(basename $bin)"_${app}
  txt_prefix="logs/$(basename $bin)_${app}_txt"
  for i in `seq 8 2 20 `; do
    grep $(( 2 ** i )) ${prefix}_${i}.log \
      | grep Blocking | head -n 1 \
      | ./scripts/fig3/transpose.sh \
      |  grep -v Blocking | grep -v $(( 2 ** i )) | tee ${txt_prefix}_$i.txt;

      echo "" | tee -a  ${txt_prefix}_$i.txt ;

    grep $(( 2 ** i )) ${prefix}_${i}.log \
      | grep Baseline | head -n 1 \
      | ./scripts/fig3/transpose.sh \
      | awk '{if(NR!=4) {print}}' \
      | grep -v Baseline | grep -v $(( 2 ** i )) | tee -a ${txt_prefix}_$i.txt ;

      echo "" ;
  done;
  paste $( ls -1 ${txt_prefix}* | sort -g ) | tee logs/results_${bin}_${app}.txt
done