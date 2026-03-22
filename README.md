# Reproducibility Guide for 2D IIM-IBAMR Simulations

## Overview

This repository contains the minimum materials needed to reproduce the 2D immersed interface method (IIM) simulations used in this work and to inspect the isolated-cylinder validation case.

Two IBAMR versions are relevant:

- Validation case: IBAMR `0.14.0`
- Present production simulations: IBAMR `0.16.0`

The compilation workflow described below has been checked with IBAMR `0.16.0` on both a local workstation and an HPC environment.

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
|   |-- input2d
|   |-- main2d
|   |-- post.py
|   |-- example.cpp
|   |-- example.o
|   |-- example.d
|   |-- CD_1.curve
|   |-- CL_1.curve
|   |-- CDisplacement_1.curve
|   |-- CTD_1.curve
|   |-- CTL_1.curve
|   |-- simulation_log.txt
|   `-- computation_results1_101_101_5.csv
|-- LICENSE
`-- README.md
```

### What Each Folder Contains

#### `example_IC_compile/`

Build directory for the 2D solver.

- `example.cpp`: customized 2D IBAMR/IIM solver source
- `CMakeLists.txt`: CMake build script
- `input2d`: example input file used during development and testing

#### `IC_mesh/`

Mesh assets for the validation case.

- `cylinder40.e`: Exodus mesh file referenced by the case input

#### `IC_VIV_Ur_5_data/`

Archived validation case for the isolated vibrating cylinder at `Ur = 5`.

This folder includes:

- the executable `main2d`
- the runtime input file `input2d`
- the post-processing script `post.py`
- archived build artifacts (`example.cpp`, `example.o`, `example.d`)
- force, displacement, and coefficient histories (`*.curve`)
- summarized post-processed results in `computation_results1_101_101_5.csv`
- runtime log information in `simulation_log.txt`

## Minimum Files Needed

### To Build the Solver

You need:

- `example_IC_compile/example.cpp`
- `example_IC_compile/CMakeLists.txt`
- a working IBAMR installation
- MPI compiler wrappers such as `mpicc` and `mpicxx`
- CMake

### To Run the Validation Case

You need:

- `IC_VIV_Ur_5_data/main2d`
- `IC_VIV_Ur_5_data/input2d`
- `IC_mesh/cylinder40.e`

The input file already points to the mesh through the relative path:

```text
MESH_FILENAME = "../IC_mesh/cylinder40.e"
```

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

## Run the Validation Case

### Option 1: Use the archived executable

The repository already includes `IC_VIV_Ur_5_data/main2d`, so you can run the case directly if that executable is compatible with your machine.

```bash
cd IC_VIV_Ur_5_data
./main2d input2d
```

### Option 2: Rebuild and replace the executable

If you want to rebuild the solver locally, copy the newly compiled executable into the case folder:

```bash
cp example_IC_compile/build/main2d IC_VIV_Ur_5_data/
```

Then run:

```bash
cd IC_VIV_Ur_5_data
./main2d input2d
```

### Parallel MPI run

```bash
cd IC_VIV_Ur_5_data
mpirun -np <NPROC> ./main2d input2d
```

Replace `<NPROC>` with the number of MPI processes you want to use. On some HPC systems, the executable may instead be launched through `srun` or another scheduler-integrated command.

## Main Outputs

The validation case folder contains these key outputs:

- `CD_1.curve`: drag-force history
- `CL_1.curve`: lift-force history
- `CDisplacement_1.curve`: cylinder displacement history
- `CTD_1.curve`: drag coefficient history
- `CTL_1.curve`: lift coefficient history
- `simulation_log.txt`: simulation timing log
- `computation_results1_101_101_5.csv`: processed summary generated by `post.py`

These files are useful for checking the force response, displacement amplitude, and post-processed quantities.

## Typical Workflow

1. Install IBAMR by following the official IBAMR build guide.
2. Build the solver in `example_IC_compile/`.
3. Copy the resulting `main2d` into `IC_VIV_Ur_5_data/` if you want to replace the archived executable.
4. Confirm that `IC_VIV_Ur_5_data/input2d` still points to `../IC_mesh/cylinder40.e`.
5. Run the case in serial or through MPI.
6. Inspect the generated `*.curve` files, `simulation_log.txt`, and the CSV summary.
7. Run `post.py` if you want to regenerate processed results.

## Reproducibility Notes

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

## Suggested Citation Practice

If this repository is shared as supplementary material or mirrored elsewhere, keep the folder names unchanged and include this README together with the original source, input file, mesh, and archived case data.
