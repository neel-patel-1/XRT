#!/bin/bash
make exe_time
sizes=( `seq 8 2 20` );

bin=exe_time
prefix="logs/$(basename $bin)"


for i in ${sizes[@]}; do
  size=$(( 2 ** ${i} ))
  sudo stdbuf -o0 ./$bin -i 100 -t 100 -a  0 -s $(( 2 ** ${i} )) -l -2 | tee -a ${prefix}_${i}.log ;
done