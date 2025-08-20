# Turek–Schäfer Benchmark Timestepping Solver

This repository provides a minimal working example of a timestepping solver for the Turek–Schäfer 2D-1 flow-around-cylinder benchmark. The solver loads a steady state (Re = 50) computed by the DefCon script and performs time integration to investigate unsteady flow behavior.

## Usage

Run the script with:

`python timestepping_solver_turek.py`

## Results

The solver outputs time-dependent velocity and pressure fields, along with computed traction forces on the obstacle. Example outputs are saved in the `output/` folder for plotting or further analysis (e.g., Paraview).

## Performance

Serial runtime depends on the number of time steps and is generally quick for the chosen Reynolds number. Parallel execution via `mpirun -n` is supported.
