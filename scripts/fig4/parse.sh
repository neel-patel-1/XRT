#!/bin/bash

PAYLOAD_SIZES=( 256 1024 4096 16384 65536 262144 1048576  )

echo "dsa memcpy"
echo "L2D L2C LLC DRAM"

for j in "${PAYLOAD_SIZES[@]}"
do
  for i in logs/placement_dsa_3_${j}_cstate_*;
  do
    grep Block $i;
  done | awk '{printf("%s ", $5);} END{printf( "\n")}'
done
