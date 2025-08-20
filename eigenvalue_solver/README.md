# Eigenvalues of the LPT Operator

Minimal working example that computes eigenvalues of the Linearized Poincaré–Steklov (LPT) operator. The script loads steady-state results from DefCon and computes the `eigs_nb` smallest eigenvalues. Only steady state for Re = 50 (first Hopf bifurcation) is included as precomputed example.

## Usage

Run the script with:

`python eigenvalue_solver_turek.py`

## Results

The computed eigenpairs are saved in folders of the form:

`eigen_output/Umax={Umax}/`

and visualized as plots in the complex plane.


