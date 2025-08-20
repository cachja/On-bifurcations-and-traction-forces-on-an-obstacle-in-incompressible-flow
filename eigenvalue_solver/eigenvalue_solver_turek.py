import firedrake as fd
from slepc4py import SLEPc
import os
import numpy as np
import matplotlib.pyplot as plt


def read_hdf(name):
    with fd.CheckpointFile(name+'.h5', 'r') as afile:
        mesh = afile.load_mesh()
        z = afile.load_function(mesh, "solution")
    return z


def my_eig_solve(U, Umax, branch, eigs_nb=10):
    # Define the Linearized Eigenvalue Problem
    W = U.function_space()
    u, p = fd.TrialFunctions(W)
    u_, p_ = fd.TestFunctions(W)
    u_base, p_base = U.split()

    # Linearized operator around the base flow
    nu = fd.Constant(0.001)
    op_A = (fd.inner(fd.grad(u) * u_base, u_) * fd.dx
            + fd.inner(fd.grad(u_base) * u, u_) * fd.dx
            + 2 * nu * fd.inner(fd.sym(fd.grad(u)), fd.grad(u_)) * fd.dx
            - fd.div(u_) * p * fd.dx
            - p_ * fd.div(u) * fd.dx)

    # Mass matrix
    m = fd.inner(u, u_) * fd.dx

    # Set up and Solve the Eigenvalue Problem
    # Zero BC
    bc_inlet = fd.DirichletBC(W.sub(0), (0, 0), 1)
    bc_cylinder = fd.DirichletBC(W.sub(0), (0, 0), 5)
    bc_walls = fd.DirichletBC(W.sub(0), (0, 0), 2)
    bc_outlet = fd.DirichletBC(W.sub(0).sub(1), 0, 3)
    bcs = [bc_inlet, bc_cylinder, bc_walls, bc_outlet]

    # Get matrices handles
    A = fd.assemble(op_A, bcs=bcs)
    M = fd.assemble(m)

    # Create the SLEPc eigensolver
    eigensolver = SLEPc.EPS().create()
    eigensolver.setOperators(A.petscmat, M.petscmat)
    eigensolver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

    # Krylovschur solver
    # Shift-and-Invert to target smallest magnitude
    eigensolver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    st = eigensolver.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
    eigensolver.setTarget(0.0)  # Focus on eigenvalues near zero
    eigensolver.setDimensions(eigs_nb)  # Number of eigenvalues to compute

    # Update solver
    eigensolver.setFromOptions()

    # Solve the eigenvalue problem
    eigensolver.solve()

    # Extract eigenvalues and eigenvectors for printing
    nconv = eigensolver.getConverged()
    print("Number of converged eigenvalues:", nconv)

    # Initialize lists to hold the real and imaginary parts of the eigenvalues
    real_parts = []
    imaginary_parts = []

    # Loop through each converged eigenvalue
    for i in range(nconv):
        eigenvalue = eigensolver.getEigenvalue(i)
        real_parts.append(eigenvalue.real)        # Real part of the eigenvalue
        # Imaginary part of the eigenvalue
        imaginary_parts.append(eigenvalue.imag)

    # There is somehow mismatch in the sign of the eigenvalues
    real_parts = -np.array(real_parts)

    # Pair eigenvalues with their original indices
    indexed_eigenvalues = [(i, real, imag) for i, (real, imag) in enumerate(
        zip(real_parts, imaginary_parts))]

    # Sort based on the real part (from largest to smallest)
    indexed_eigenvalues_sorted = sorted(
        indexed_eigenvalues, key=lambda x: x[1], reverse=True)

    # Write to a text file in the desired format
    if not os.path.exists(f"eigen_output/Umax={Umax}/"):
        os.makedirs(f"eigen_output/Umax={Umax}/")
    output_path_with_indices = f'eigen_output/Umax={Umax}/eigenvalues_branch{branch}.txt'
    # Check unsteable eigenpairs s.t. Re>=0 and Im>0
    unstable_i = []
    with open(output_path_with_indices, 'w') as file:
        file.write("i\treal\timaginary\n")
        for i, (orig_index, real, imag) in enumerate(indexed_eigenvalues_sorted):
            file.write(f"{orig_index}\t{real:.6f}\t{imag:.6f}\n")
            if real > 0 and imag >= 0:
                print(real, imag)
                unstable_i.append(i)
    print(f"{unstable_i=}")

    # Plotting of all converged eigenvalues
    plt.figure(figsize=(8, 6))
    plt.plot(real_parts, imaginary_parts, '+', markersize=10, color='blue')
    plt.axhline(0, color='gray', lw=0.5)         # Add horizontal line at y=0
    plt.axvline(0, color='gray', lw=0.5)         # Add vertical line at x=0
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Eigenvalues in the Complex Plane')
    plt.grid(True)
    plt.savefig(f"eigen_output/Umax={Umax}/eigen_plot_branch{branch}.png")
    plt.clf()

    # Collect the unstable eigenmodes --- might be plotted
    vp_ri = fd.Function(W)
    eigenmode_set = []
    for i in unstable_i:
        print(f"{i=}")
        real = indexed_eigenvalues_sorted[i][1]
        imag = indexed_eigenvalues_sorted[i][2]
        vp_r = vp_ri.copy(deepcopy=True)
        vp_i = vp_ri.copy(deepcopy=True)
        with vp_r.dat.vec as vr_vec, vp_i.dat.vec as vi_vec:
            eigensolver.getEigenvector(
                indexed_eigenvalues_sorted[i][0], vr_vec, vi_vec)
        eigenmode_set.append(
            {'i': i, 'vp_r': vp_r, 'vp_i': vp_i, 'real': real, 'imag': imag})
    return eigenmode_set


if __name__ == "__main__":
    branch = 0  # Baseline branch of the steady solutions
    z = read_hdf(f"Umax=7.600000000000005e-01/solution-{branch}")
    Umax = 0.76  # Re=50
    print(f"Re = {Umax/3.*2.*100.}")
    eigenmodes = my_eig_solve(z, Umax, branch)
