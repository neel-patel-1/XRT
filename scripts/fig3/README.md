### Fig 3 Execution Time Breakdown of Three Phase Workloads Accelerated/Unaccelerated

```sh
# from root of repo
./scripts/fig3/run.sh
./scripts/fig3/parse.sh
```

* Prints tabulated execution time breakdown
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