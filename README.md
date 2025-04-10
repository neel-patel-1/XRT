# XRT: An Accelerator-Aware Runtime For Accelerated Chip Multiprocessors
* Some steps have already been completed for AE. AE can refer to **bolded** instructions to aid in the reproduction process.
* For **AE's** convenience, we have provided a jupyter lab session running on the host machine we used to generate the figures in XRT. **Please comment on the AE HotCRP site to notify us that AE is ready and We will provide the link to access the jupyter lab and the login/password**.
* **To access the jupyter lab, and begin reproducing start with the instructions below:**
1) log in to the jupyter lab by connecting to the provided url using your browser and logging in with the provided username/password.
2) upon logging in, you will see the launcher screen
3) select terminal under the options labelled "other" to open a terminal
4) type `cd XRT` to enter the root directory of the XRT repository (this will be the only directory in your home directory)
5) Continue with the instructions below

### Build dependencies and configure
* Below are instructions to build all dependencies and configure INTEL(R) XEON(R) PLATINUM 8571N running Ubuntu 24.04 to run XRT

* **Building has already been completed for AE. AE can skip to configuration step generation (next).**
* Build dependecies
```sh
./build_all.sh
```

* Configure IAA && DSA, download packages dependencies, and activate python environment
```sh
source ./scripts/setup.sh
```

### Figure generation
* The following instructions detail how to generate and view the figures from the paper

### Figure 3
* Below loads a worker-centric system and a dispatcher-centric system running the deserialize-decompress-hash workload. Then plots the resulting slowdown in `worker_vs_dispatcher.png`
* **Notes For AE**: All scripts should be run from the root directory of the XRT repository -- they will not work otherwise. To view the figure, use the FileExplorer on the left-hand-side of the UI to open the `worker_vs_dispatcher.png` file in the jupyter lab. If the plot does not appear right away, either wait a few seconds or click the refresh button in the file explorer.
```sh
./scripts/fig3/run.sh
./scripts/fig3/plotter.sh
```

### Figure 4
* Below plots loads a worker-centric system running under three different configurations (NoAcceleration, Block, and RR\_Worker) running the deserialize-decompress-hash workload. Then plots the resulting slowdown in `blocking_overhead.png`
* **For AE**: To view the figure, use the FileExplorer on the left-hand-side of the UI to open the `blocking_overhead.png` file in the jupyter lab
```sh
./scripts/fig4/run.sh
./scripts/fig4/plotter.sh
```

### Figure 6
* Below compares each system configuration (NoAcceleration, Block, RR\_Worker, and XRT) on each three-phase workload. Then plots the resulting performance comparison in `ddh.png`, `dmdp.png`, `dg.png`, `mg.png`, `ufh.png`, and `mmp.png`.
* **For AE**: To view the figure, use the FileExplorer on the left-hand-side of the UI to open the `ddh.png`, `dmdp.png`, `dg.png`, `mg.png`, `ufh.png`, and `mmp.png` files in the jupyter lab
```sh
./scripts/fig6/runner.sh
./scripts/fig6/plotter.sh
```
