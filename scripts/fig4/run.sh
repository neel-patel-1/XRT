#!/bin/bash

# sudo python3 scripts/accel_conf.py --load=configs/iaa-1n1d8e1w128q-s-n2.conf

bin=exe_time
make $bin -j

PAYLOAD_SIZES=( 256 1024 4096 16384 65536 262144 1048576 )
PLACEMENTS=( 0 1 2 3 )

dsa_app=8
dsa_opcodes=( 3 4 )

iaa_app=9
iaa_opcodes=( 66 )


for opcode in "${dsa_opcodes[@]}";
do
  prefix="logs/placement_dsa_${opcode}"
  for query_size in "${PAYLOAD_SIZES[@]}";
  do
    for cstate in "${PLACEMENTS[@]}";
    do
      log="${prefix}_${query_size}_cstate_${cstate}.log"
      sudo \
        ./$bin -t 100 -a $dsa_app -o $opcode -s $query_size -l 5 -p $cstate \
        | tee ${log}
    done
  done
done