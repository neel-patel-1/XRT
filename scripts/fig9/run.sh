#!/bin/bash

SIZES=( 256 1024 4096 16384 $(( 64 * 1024 )) $(( 256 * 1024 )) $(( 1024 * 1024 )) )
bin=./main
reqs=100

make $bin CXXFLAGS="-O3 -DEXETIME"  -j
prefix="logs/$(basename $bin)"
for size in "${SIZES[@]}"; do
  sudo $bin -g -b $size -t ${reqs}  -l -2 | tee ${prefix}_noacc_$size.log ;
  sudo $bin -b $size -t ${reqs}  -l -2 | tee ${prefix}_dummy_$size.log ;
  sudo $bin -s 0 -b $size -t ${reqs}  -l -2 | tee ${prefix}_syncdemote_$size.log ;
  sudo $bin -s 1 -b $size -t ${reqs}  -l -2 | tee ${prefix}_syncpref_$size.log ;
  sudo $bin -s 2 -b $size -t ${reqs}  -l -2 | tee ${prefix}_syncboth_$size.log ;
  paste ${prefix}_noacc_$size.log ${prefix}_dummy_$size.log ${prefix}_syncdemote_$size.log ${prefix}_syncpref_$size.log  \
  ${prefix}_syncboth_$size.log | tee ${prefix}_$size.log
done
