#!/bin/bash

PAYLOAD_SIZES=( 256 1024 4096 16384 65536 262144 1048576  )

echo ""
echo "gpcore memcpy"
echo "L2D L2C LLC DRAM"

for j in "${PAYLOAD_SIZES[@]}"
do
  for i in logs/placement_dsa_3_${j}_cstate_*;
  do
    grep Baseline $i;
  done | awk '{printf("%s ", $5);} END{printf( "\n")}'
done

echo ""
echo "dsa memcpy"
echo "L2D L2C LLC DRAM"

for j in "${PAYLOAD_SIZES[@]}"
do
  for i in logs/placement_dsa_3_${j}_cstate_*;
  do
    grep Block $i;
  done | awk '{printf("%s ", $5);} END{printf( "\n")}'
done

echo ""
echo "gpcore memfill"
echo "L2D L2C LLC DRAM"

for j in "${PAYLOAD_SIZES[@]}"
do
  for i in logs/placement_dsa_4_${j}_cstate_*;
  do
    grep Baseline $i;
  done | awk '{printf("%s ", $5);} END{printf( "\n")}'
done

echo ""
echo "dsa memfill"
echo "L2D L2C LLC DRAM"

for j in "${PAYLOAD_SIZES[@]}"
do
  for i in logs/placement_dsa_4_${j}_cstate_*;
  do
    grep Block $i;
  done | awk '{printf("%s ", $5);} END{printf( "\n")}'
done
