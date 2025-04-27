#!/bin/bash

bin=exe_time
prefix="logs/$(basename $bin)"
txt_prefix="logs/$(basename $bin)_ddh"
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
paste $( ls -1 ${txt_prefix}* | sort -g )


#for i in `seq 8 2 20 `; do  grep $(( 2 ** i )) ddh_exe_time.log | grep Blocking | head -n 1 | scripts/transpose.sh |  grep -v Blocking | grep -v $(( 2 ** i )) | tee $(( 2 ** i ))_ddh.txt ; echo "" | tee -a  $(( 2 **i))_ddh.txt;  grep $(( 2 ** i )) ddh_exe_time.log | grep Baseline | head -n 1 | scripts/transpose.sh | awk '{if(NR!=4) {print}}' | grep -v Baseline | grep -v $(( 2 ** i )) | tee -a $((2 **i))_ddh.txt ; echo "" ; done; paste $( ls -1 *_ddh.txt | sort -g )

