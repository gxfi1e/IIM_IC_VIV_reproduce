# Reproducibility Guide for 2D IIM-IBAMR Simulations

## Overview

This repository contains the minimum materials needed to reproduce the 2D immersed interface method (IIM) simulations used in this work and to inspect the isolated-cylinder validation case.

Two IBAMR versions are relevant:

- Validation case: IBAMR `0.14.0`
- Present production simulations: IBAMR `0.16.0`

The build workflow described below has been checked with IBAMR `0.16.0` on both a local workstation and an HPC environment.

Official IBAMR resources:

- Downloads: <https://ibamr.github.io/downloads>
- Build guide: <https://ibamr.github.io/building>
- AutoIBAMR helper project: <https://github.com/IBAMR/autoibamr>

## Repository Layout

```text
IIM_IC_VIV_reproduce/
|-- example_IC_compile/
|   |-- CMakeLists.txt
|   |-- example.cpp
|   `-- input2d
|-- IC_mesh/
|   `-- cylinder40.e
|-- IC_VIV_Ur_5_data/
|   |-- example.cpp
|   |-- input2d
|   |-- main2d
|   `-- previous_data/
|       |-- post.py
|       |-- CD_1.curve
|       |-- CL_1.curve
|       |-- CDisplacement_1.curve
|       |-- CTD_1.curve
|       |-- CTL_1.curve
|       |-- simulation_log.txt
|       |-- computation_results1_101_101_5.csv
|       |-- example.o
|       `-- example.d
|-- LICENSE
`-- README.md
```

## Folder Roles

### `example_IC_compile/`

Build directory for the 2D solver.

- `example.cpp`: customized 2D IBAMR/IIM solver source
- `CMakeLists.txt`: CMake build script
- `input2d`: example input file used during development and testing

### `IC_mesh/`

Mesh assets for the validation case.

- `cylinder40.e`: Exodus mesh file referenced by the case input

### `IC_VIV_Ur_5_data/`

Starter files for reproducing the `Ur = 5` validation case.

- `main2d`: executable that can be used as a starting point for a rerun
- `input2d`: runtime input file for a fresh run
- `example.cpp`: source snapshot for reference
- `previous_data/`: preserved snapshot of the author's archived results

### `IC_VIV_Ur_5_data/previous_data/`

Read-only reference data from the archived validation run.

This folder contains:

- output histories such as `CD_1.curve`, `CL_1.curve`, `CDisplacement_1.curve`, `CTD_1.curve`, and `CTL_1.curve`
- the archived timing log `simulation_log.txt`
- the processed summary `computation_results1_101_101_5.csv`
- the post-processing script `post.py`
- archived build artifacts such as `example.o` and `example.d`

## Important Usage Note

`IC_VIV_Ur_5_data/previous_data/` is an archived snapshot and should be kept unchanged.

For a clean separation between archived results and new results, do not run the solver inside `IC_VIV_Ur_5_data/previous_data/`, and preferably do not run directly inside `IC_VIV_Ur_5_data/` either. The recommended approach is to create a separate working directory for your own rerun.

## Minimum Files Needed

### To Build the Solver

You need:

- `example_IC_compile/example.cpp`
- `example_IC_compile/CMakeLists.txt`
- a working IBAMR installation
- MPI compiler wrappers such as `mpicc` and `mpicxx`
- CMake

### To Run a Fresh Validation Case

You need:

- an executable `main2d`
- an input file `input2d`
- `IC_mesh/cylinder40.e`

For convenience, the repository already provides starter copies in `IC_VIV_Ur_5_data/`.

The supplied input file uses the mesh path:

```text
MESH_FILENAME = "../IC_mesh/cylinder40.e"
```

This path remains valid if you create your new run directory as a sibling of `IC_mesh/`, as recommended below.

## Software Requirements

This example depends on CMake, MPI, and IBAMR.

- CMake `>= 3.15`
- MPI compiler wrappers: `mpicc`, `mpicxx`
- An IBAMR installation that provides `IBAMRConfig.cmake`

Notes:

- On clusters, module names depend on the local environment.
- `IBAMR_DIR` should point to the directory that contains `IBAMRConfig.cmake`.
- The provided `CMakeLists.txt` requires IBAMR `0.16.0`.

If needed, you can search for the IBAMR CMake package with:

```bash
find /usr /usr/local /opt $HOME /ibex -name IBAMRConfig.cmake 2>/dev/null
find /usr /usr/local /opt $HOME /ibex -name ibamr-config.cmake 2>/dev/null
```

If the result is, for example:

```bash
/home/user/autoibamr/packages/IBAMR-0.16.0/lib64/cmake/ibamr/IBAMRConfig.cmake
```

then use:

```bash
-DIBAMR_DIR=/home/user/autoibamr/packages/IBAMR-0.16.0/lib64/cmake/ibamr
```

## Build the Solver

From the repository root:

```bash
cd example_IC_compile
mkdir -p build
cd build

cmake .. \
  -DCMAKE_C_COMPILER=$(which mpicc) \
  -DCMAKE_CXX_COMPILER=$(which mpicxx) \
  -DIBAMR_DIR=/path/to/IBAMR/lib64/cmake/ibamr

cmake --build . -j
```

After a successful build, the executable `main2d` should appear in `example_IC_compile/build/`.

If your system uses environment modules, load them before configuring:

```bash
module load cmake/<cmake-version>
module load openmpi/<mpi-version>
```

## Recommended Rerun Workflow

To keep your new outputs separate from the archived reference snapshot, create a new working directory at the repository root.

### Step 1: create a clean run folder

For example:

```bash
mkdir IC_VIV_Ur_5_run
cp IC_VIV_Ur_5_data/main2d IC_VIV_Ur_5_run/
cp IC_VIV_Ur_5_data/input2d IC_VIV_Ur_5_run/
cp IC_VIV_Ur_5_data/previous_data/post.py IC_VIV_Ur_5_run/
```

The first two files are required. Copying `post.py` is optional, but convenient if you want to regenerate processed results after the run.

### Step 2: optionally replace the executable with your rebuilt one

If you compiled a new solver locally, replace the starter executable:

```bash
cp example_IC_compile/build/main2d IC_VIV_Ur_5_run/main2d
```

### Step 3: verify the mesh path

If your run directory is created exactly as shown above, the existing line in `input2d`

```text
MESH_FILENAME = "../IC_mesh/cylinder40.e"
```

can stay unchanged.

### Step 4: run the case

Serial run:

```bash
cd IC_VIV_Ur_5_run
./main2d input2d
```

Parallel MPI run:

```bash
cd IC_VIV_Ur_5_run
mpirun -np <NPROC> ./main2d input2d
```

Replace `<NPROC>` with the number of MPI processes you want to use. On some HPC systems, the executable may instead be launched through `srun` or another scheduler-integrated command.

## Archived Reference Outputs

The archived validation outputs are stored in `IC_VIV_Ur_5_data/previous_data/`.

Key files include:

- `CD_1.curve`: drag-force history
- `CL_1.curve`: lift-force history
- `CDisplacement_1.curve`: cylinder displacement history
- `CTD_1.curve`: drag coefficient history
- `CTL_1.curve`: lift coefficient history
- `simulation_log.txt`: simulation timing log
- `computation_results1_101_101_5.csv`: processed summary generated from the archived run

Use these files as reference data for comparison with the outputs generated in your own run directory.

## Post-Processing

The archived post-processing script is stored at:

```text
IC_VIV_Ur_5_data/previous_data/post.py
```

The script reads `input2d` and output files from the current working directory. Therefore, if you want to post-process a fresh rerun, copy `post.py` into your new run directory and execute it there after the simulation finishes.

## Typical Workflow

1. Install IBAMR by following the official IBAMR build guide.
2. Build the solver in `example_IC_compile/`, or use the provided starter executable.
3. Create a clean working directory such as `IC_VIV_Ur_5_run/`.
4. Copy `main2d` and `input2d` into that new directory.
5. Optionally copy `post.py` from `IC_VIV_Ur_5_data/previous_data/`.
6. Run the case in serial or through MPI.
7. Compare your newly generated outputs against the archived files in `IC_VIV_Ur_5_data/previous_data/`.

## Reproducibility Notes

- Do not modify or overwrite files inside `IC_VIV_Ur_5_data/previous_data/` if you want to preserve the archived snapshot.
- If `input2d` is edited to use absolute paths, those paths must be updated on a new machine.
- The validation case in this package was originally run with IBAMR `0.14.0`.
- The present production simulations reported in the manuscript were run with IBAMR `0.16.0`.
- The current build procedure in this repository has been checked with IBAMR `0.16.0`.
- Exact agreement with published results requires the same source, input settings, mesh, and software environment.

## Origin and References

The present 2D solver was customized from IBAMR IIM example 4:

- <https://github.com/IBAMR/IBAMR/tree/master/examples/IIM/ex4>

For the immersed-interface rigid-body formulation, see:

- E. M. Kolahdouz et al., *A sharp interface Lagrangian-Eulerian method for rigid-body fluid-structure interaction*, **Journal of Computational Physics**, 2021. DOI: <https://doi.org/10.1016/j.jcp.2021.110442>

For the isolated-cylinder validation case used here, see:

- M. Gong and B. Dally, *Dynamic response of a cylinder-plate-cylinder system coupling wake-induced and interference effects*, **Physics of Fluids**, 2025, 37(10): 103602. DOI: <https://doi.org/10.1063/5.0289251>
