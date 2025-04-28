#!/bin/bash

# sudo python3 scripts/accel_conf.py --load=configs/iaa-1n1d8e1w128q-s-n2.conf

bin=exe_time
make $bin -j

PAYLOAD_SIZES=( 256 1024 4096 16384 65536 262144 1048576 )
PLACEMENTS=( 0 1 2 3 )

dsa_app=8
DSA_MEMMOVE=3
DSA_MEMFILL=4
dsa_opcodes=( ${DSA_MEMMOVE} ${DSA_MEMFILL} )

iaa_app=9
IAA_DECOMPRESS=66
iaa_opcodes=( ${IAA_DECOMPRESS} )


for opcode in "${dsa_opcodes[@]}";
do
  prefix="logs/placement_dsa_${opcode}"
  for query_size in "${PAYLOAD_SIZES[@]}";
  do
    for cstate in "${PLACEMENTS[@]}";
    do
      log="${prefix}_${query_size}_cstate_${cstate}.log"
      sudo \
        ./$bin -i 1 -t 100 -a $dsa_app -o $opcode -s $query_size -l -2 -p $cstate -q -j \
        | tee ${log}
    done
  done
done

for opcode in "${iaa_opcodes[@]}";
do
  prefix="logs/placement_iaa_${opcode}"
  for query_size in "${PAYLOAD_SIZES[@]}";
  do
    for cstate in "${PLACEMENTS[@]}";
    do
      log="${prefix}_${query_size}_cstate_${cstate}.log"
      sudo \
        ./$bin -i 1 -t 100 -a $iaa_app -o $opcode -s $query_size -l -2 -p $cstate -q -j \
        | tee ${log}
    done
  done
done