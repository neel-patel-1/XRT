#!/bin/bash

python3 scripts/fig6/plot.py logs/main_xrt-2600DDH-0DMDP-0DG-0MG-0UFH-0MMP-Yield-25w-128c-0dist-1024-1048576bimodal-2type-all.txt ddh.png

python3 scripts/fig6/plot.py logs/main_xrt-0DDH-2600DMDP-0DG-0MG-0UFH-0MMP-Yield-25w-128c-0dist-1024-1048576bimodal-2type-all.txt dmd.png

python3 scripts/fig6/plot.py logs/main_xrt-0DDH-0DMDP-2600DG-0MG-0UFH-0MMP-Yield-25w-128c-0dist-1024-1048576bimodal-2type-all.txt dc.png

python3 scripts/fig6/plot.py logs/main_xrt-0DDH-0DMDP-0DG-2600MG-0UFH-0MMP-Yield-25w-128c-0dist-1024-1048576bimodal-2type-all.txt mc.png

python3 scripts/fig6/plot.py logs/main_xrt-0DDH-0DMDP-0DG-0MG-2600UFH-0MMP-Yield-25w-128c-0dist-1024-1048576bimodal-2type-all.txt ufh.png

python3 scripts/fig6/plot.py logs/main_xrt-0DDH-0DMDP-0DG-0MG-0UFH-2600MMP-Yield-25w-128c-0dist-1024-1048576bimodal-2type-all.txt mmp.png

