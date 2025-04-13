### Fig 9 Speedup from Sync/Concurrent Demotion

```sh
# from root of repo
./scripts/fig9/run.sh
./scripts/fig9/parse.sh

# Output is breakdown of end-to-end execution time in each phase including synchronous prefetching (MM_PREFETCH) and synchronous demotion (CLDEMOTE)
# For performance of concurrent optimizations, subtract synchronous prefetch and/or synchronous demotion
```
