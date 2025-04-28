### Fig 3 Execution Time Breakdown of Three Phase Workloads Accelerated/Unaccelerated

```sh
# from root of repo
./scripts/fig4/run.sh
./scripts/fig4/parse.sh
```

* Memory Hierarchy Placement Stats for gpCore/axCore memcpy/memfill/decompress organized in ascending payload size from 256B to 1MB
```sh
gpcore decompress
L2D L2C LLC DRAM
9040 6058 10141 13024
61717 49906 58571 34656
119111 228092 134331 128609
441137 472344 438695 466036
1631840 1583595 1630084 1774563
6471800 6403107 6470491 6358165
25352952 25238632 25340994 25278334
```