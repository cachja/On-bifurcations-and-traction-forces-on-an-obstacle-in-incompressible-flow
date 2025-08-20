# Benchmarking Pointwise Traction Computation Methods on Turek–Schäfer Benchmark

This repository implements benchmark tests for pointwise traction computations using the Turek–Schäfer 2D-1 flow-around-cylinder benchmark.

## Cases

- **NS**: Laminar Navier–Stokes case, Re = 20  
- **S**: Stokes case, Re = 0  

## Usage

Run both benchmarks with `make`, or individual cases by calling the corresponding target.

## Results & Performance

Key results are saved in the `EOCplot` folder as:

- `convergence_curves_NS.pdf`  
- `convergence_curves_S.pdf`  

Serial runtime: ~2–3 minutes (max DoF 520,000 with `MAX_REF = 5`).  
Parallel runs are supported, e.g. `mpirun -n 4 make` for more refinement levels.  
When running in parallel, ensure that the `EOCplot` and `EOCdata` folders exist beforehand.