#!/bin/bash
[ ! -f env/bin/activate ] && echo "Please run setup.sh first" && exit 1
source env/bin/activate
python3 scripts/fig3/plot.py