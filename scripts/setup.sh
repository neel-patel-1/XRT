#!/bin/bash
sudo python3 scripts/accel_conf.py --load=scripts/confs/idxd-2n8d64e8w-s-n1-n2.conf
[ ! -f env/bin/activate ] && \
  python3 -m venv env && \
  pip install -r requirements.txt
source env/bin/activate
