#!/bin/bash

bin=exe_time
make $bin -j
APPS=( 0 1 2 7 10 11 )
sizes=( `seq 8 2 20` );

for app in "${APPS[@]}"; do
  prefix="logs/$(basename $bin)"_${app}
  for i in ${sizes[@]}; do
    size=$(( 2 ** ${i} ))
    sudo stdbuf -o0 ./$bin -i 100 -t 100 -a $app -s $(( 2 ** ${i} )) -l -2 | tee -a ${prefix}_${i}.log ;
  done
done