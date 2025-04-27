### Fig 3 Execution Time Breakdown of Three Phase Workloads Accelerated/Unaccelerated

```sh
# from root of repo
./scripts/fig3/run.sh
./scripts/fig3/parse.sh
```

* Exetime breakdowns for each app are in logs/results_exe_time_<appno>.txt
* appno:
```
APPS=( 0 1 2 7 10 11 )
0 - ddh
1 - dc
2 - mc
7 - dmd
10 - mmp
11 - ufh
```

* Rows are ordered/formatted:
```
Pre-Processing (Block&Wait)
Offload Tax (Block&Wait)
Accelerable (Block&Wait)
Post-Processing (Block&Wait)

Pre-Processing (NoAcceleration)
Accelerable (NoAcceleration)
Post-Processing (NoAcceleration)
```