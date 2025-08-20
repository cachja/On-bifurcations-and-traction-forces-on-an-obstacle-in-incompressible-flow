# Firedrake installation with PETSc required
import firedrake as fd
import os
# Matplotlib with local LaTeX installation required
import matplotlib.pyplot as plt
import numpy as np

comm = fd.COMM_WORLD
rank = comm.Get_rank()

def get_boundary_values(V, mesh, bndry_num, fun, fun_index):
    # Get values vertex by vertex on a given boundary
    def create_vertex_values(V, bndry_num):
        dofs_in_vertex = {}
        # Use DirichletBC to extract DoFs on the boundary
        boundary_dofs = fd.DirichletBC(V, 0, bndry_num).nodes

        # Get coordinates of DoFs
        mesh_coords = mesh.coordinates.dat.data  # Coordinates of the mesh

        for dof in boundary_dofs:
            coord = tuple(mesh_coords[dof])
            dofs_in_vertex[dof] = {
                "coordinates": coord,
                "dof": dof
            }
        return dofs_in_vertex

    dofs_in_vertex = create_vertex_values(V, bndry_num)

    # Append values we are interested in
    def get_values_to_list(fun, fun_index):
        fun_values = []
        fun_coord = []

        for vertex in dofs_in_vertex.values():
            coord = vertex["coordinates"]
            # Evaluate function at coordinates
            fun_value = fun.at(coord)[fun_index]
            fun_values.append(fun_value)
            fun_coord.append(coord)

        return (fun_values, fun_coord)

    (fun_values, fun_coord) = get_values_to_list(
        fun, fun_index)

    return (fun_coord, fun_values)

def compute_solution(W,Umax):
    mesh = W.mesh()

    x = fd.SpatialCoordinate(mesh)
    inflow_profile = fd.as_vector([4.0*Umax*x[1]*(0.41-x[1])/0.41/0.41,0])

    bc_inlet = fd.DirichletBC(W.sub(0), inflow_profile, 1)
    bc_cylinder = fd.DirichletBC(W.sub(0), (0, 0), 5)
    bc_walls = fd.DirichletBC(W.sub(0), (0, 0), 2)
    bcs = [bc_inlet, bc_cylinder, bc_walls]
    # Define unknown and test function(s)
    (v_, p_) = fd.TestFunctions(W)

    # current unknown time step
    w = fd.Function(W)
    (v, p) = fd.split(w)

    def a(v,u,nu=fd.Constant(0.001)):
        return (fd.inner(fd.grad(v)*v, u) + fd.inner(nu*fd.grad(v), fd.grad(u)))*fd.dx

    def b_form(q,v):
        return fd.inner(fd.div(v),q)*fd.dx

    F = a(v,v_) - b_form(p_,v) - b_form(p,v_)

    J = fd.derivative(F, w)

    problem=fd.NonlinearVariationalProblem(F,w,bcs,J)
    lu = {"mat_type": "aij",
          "snes_type": "newtonls",
          "snes_monitor": None,
          "snes_converged_reason": None,
          "snes_max_it": 12,
          "snes_rtol": 1e-11,
          "snes_atol": 5e-10,
          "snes_linesearch_type": "basic",
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps"}
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=lu)

    solver.solve()
    return w

def compute_traction_bndryval(U):
    # Build spaces for traction (linear problem)
    W = U.function_space()
    mesh = W.mesh()
    v, p = U.split()
    V = fd.VectorFunctionSpace(mesh, "CG", 1) # Mind the degree
    g = fd.TrialFunction(V)
    g_ = fd.TestFunction(V)
    fd.info(f"{V.dim()=}")

    # set BC
    bc_list = [1,2,3] # zero traction on bndries not touching bndry of interest
    bc_num = 5 # circle bndry - here we solve PS
    bcs_g = [fd.DirichletBC(V, fd.Constant((0,0)), i) for i in bc_list]

    # Define the bilinear form for disrete variational traction problem
    a_g = fd.inner(g , g_) * fd.ds(bc_num)

    def a(v,u,nu=fd.Constant(0.001)) :
        return (fd.inner(fd.grad(v)*v, u) + fd.inner(nu*fd.grad(v), fd.grad(u)))*fd.dx

    def b(q,v) :
        return fd.inner(fd.div(v),q)*fd.dx

    def F1(v,p,v_):
        return a(v,v_) - b(p,v_)

    L_g = F1(v,p,g_)

    # Solve linear system
    gg = fd.Function(V)
    A = fd.assemble(a_g, bcs = bcs_g)
    b = fd.assemble(L_g)

    # Dirty ident_zeros trick
    diagonal = A.petscmat.getDiagonal()
    vals = diagonal.array
    vals[vals == 0] = 1
    A.petscmat.setDiagonal(diagonal, fd.PETSc.InsertMode.INSERT_VALUES)
    
    fd.info("Solving DVND using sparse LU")
    fd.solve(A,gg,b , solver_parameters={"ksp_type": "preonly",
                                                  "pc_type": "lu",
                                                  "pc_factor_mat_solver_type": "mumps"})
    fd.info("DVND solved")

    lift = fd.assemble(-2.0/(0.2*0.2*0.1)*gg[1]*fd.ds(5))
    print(f"{lift=}")

    (drag_coords, drag_values) = get_boundary_values(V,mesh,5,gg,0)
    drag_values = np.array(drag_values).tolist()
    (lift_coords, lift_values) = get_boundary_values(V,mesh,5,gg,1)
    lift_values = np.array(lift_values).tolist()

    return (drag_coords,drag_values,lift_coords,lift_values)

def merge_data(drag_coords, drag_values_list):
    """
    Merge drag coordinates and drag values from multiple MPI ranks.

    Parameters:
    - drag_coords_list: List of lists where each list contains drag coordinates from a rank.
    - drag_values_list_list: List of lists where each list contains drag values from a rank.

    Returns:
    - merged_drag_coords: Flattened list of drag coordinates from all ranks.
    - merged_drag_values_list: List of lists where each list contains concatenated drag values for each component.
    """

    # Gather all data from all ranks
    all_drag_coords = comm.allgather(drag_coords)
    all_drag_values_list = comm.allgather(drag_values_list)

    # Flatten the list of coordinates
    merged_drag_coords = [coord for rank_coords in all_drag_coords for coord in rank_coords]

    # Merging drag_values_list
    # First initialize an empty list for the merged drag values list
    if len(drag_values_list) > 0:
        num_components = len(drag_values_list)
        merged_drag_values_list = [[] for _ in range(num_components)]

        for rank_drag_values_list in all_drag_values_list:
            for i in range(num_components):
                merged_drag_values_list[i].extend(rank_drag_values_list[i])

    else:
        # If drag_values_list is empty for some reason, just handle it gracefully
        merged_drag_values_list = []

    return merged_drag_coords, merged_drag_values_list

def calculate_cumulative_length(points):
    """
    Calculate the cumulative length of a curve given as a list of points.
    
    Parameters:
    - points: List of tuples (x, y), where each tuple represents a point on the curve.
    
    Returns:
    - A list of cumulative lengths.
    """
    def distance(p1, p2):
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    cumulative_lengths = [0]  # Starting point with length 0
    total_length = 0
    
    for i in range(1, len(points)):
        segment_length = distance(points[i-1], points[i])
        total_length += segment_length
        cumulative_lengths.append(total_length)

    return cumulative_lengths

def calculate_angles(points, center=(0.2, 0.2)):
    # Calculate the angles of points around a circle relative to the shifted center.
    cx, cy = center
    return [np.arctan2(y - cy, x - cx) for x, y in points]

def sort_and_split_data(coordinates, list_of_results, y_threshold=0.2):
    # Separate upper and lower coordinates and results based on y value
    x_upper, y_upper = [], []
    x_lower, y_lower = [], []
    
    results_upper = [[] for _ in range(len(list_of_results))]
    results_lower = [[] for _ in range(len(list_of_results))]

    for (x, y), *results in zip(coordinates, *list_of_results):
        if y <= 0.1999:
            x_lower.append(x)
            y_lower.append(y)
            for i, result in enumerate(results):
                results_lower[i].append(result)
        else:  # Upper wing boundary
            x_upper.append(x)
            y_upper.append(y)
            for i, result in enumerate(results):
                results_upper[i].append(result)
    
    # Sorting upper and lower wing data by x-coordinate
    x_upper_unsorted = x_upper
    x_lower_unsorted = x_lower
    x_upper, y_upper = zip(*sorted(zip(x_upper, y_upper)))
    x_lower, y_lower = zip(*sorted(zip(x_lower, y_lower)))
    
    # Sort each list of results according to the sorted x-coordinates
    sorted_results_upper = []
    sorted_results_lower = []
    
    for results in results_upper:
        sorted_results_upper.append([result for x, result in sorted(zip(x_upper_unsorted, results))])
    
    for results in results_lower:
        sorted_results_lower.append([result for x, result in sorted(zip(x_lower_unsorted, results))])
    
    return x_upper, y_upper, sorted_results_upper, x_lower, y_lower, sorted_results_lower

def plot_profiles_multiple(coordinates, list_of_results, labels, name, path):
    # Separate upper and lower coordinates based on y value
    x_upper, y_upper, sorted_results_upper, x_lower, y_lower, sorted_results_lower = sort_and_split_data(coordinates, list_of_results)
    print(len(x_upper))
    print(len(x_lower))

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        'axes.titlesize': 14,   # Title font size
        'axes.labelsize': 18,   # X and Y labels font size
        'xtick.labelsize': 14,  # X-axis tick labels font size
        'ytick.labelsize': 14,  # Y-axis tick labels font size
        'legend.fontsize': 12   # Legend font size
    })


    # Create the plot with a custom height ratio
    fig, axs = plt.subplots(2, 2, figsize=(6, 8), gridspec_kw={'height_ratios': [1, 6]})

    # Upper left: Upper wing shape (y vs x)
    axs[0, 0].plot(x_upper, y_upper, color='black')
    axs[0, 0].set_title("upper cylinder part")
    axs[0, 0].set_xlabel(r"x-coord. [m]")
    axs[0, 0].set_ylabel(r"y-coord. [m]")
    #axs[0, 0].set_ylim(0.190,0.245)

    # Upper right: Lower wing shape (y vs x)
    axs[0, 1].plot(x_lower, y_lower, color='black')
    axs[0, 1].set_title("lower cylinder part")
    axs[0, 1].set_xlabel(r"x-coord. [m]")
    axs[0, 1].set_ylabel(r"y-coord. [m]")
    #axs[0, 1].set_ylim(0.190,0.245)

    cumulative_lengths_upper = calculate_cumulative_length([(x, y) for x, y in zip(x_upper, y_upper)])
    cumulative_lengths_lower = calculate_cumulative_length([(x, y) for x, y in zip(x_lower, y_lower)])

    # Plot each result set in the lower left subplot
    for sorted_results, label in zip(sorted_results_upper, labels):
        axs[1, 0].plot(np.array(cumulative_lengths_upper)/0.05, sorted_results, label=fr"$Re={label}$")
    #axs[1, 0].set_title("Results for Upper Cylinder Part")
    axs[1, 0].set_xlabel(r"$\theta$ [rad]")
    axs[1, 0].set_ylabel(name)
    axs[1, 0].ticklabel_format(axis='y', style='sci',useOffset=False, scilimits=(-2,-2))
    x_ticks = np.linspace(0, np.pi, 4)  # 0, π/3, 2π/3, π
    axs[1, 0].set_xticks(x_ticks)
    axs[1, 0].set_xticklabels([r"$0$", r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$", r"$\pi$"])
    axs[1, 0].legend()

    # Plot each result set in the lower right subplot
    for sorted_results, label in zip(sorted_results_lower, labels):
        axs[1, 1].plot(np.array(cumulative_lengths_lower)/0.05, sorted_results, label=fr"$Re={label}$")
    #axs[1, 1].set_title("Results for Lower Cylinder Part")
    axs[1, 1].set_xlabel(r"$\theta$ [rad]")
    axs[1, 1].set_ylabel(name)
    axs[1, 1].ticklabel_format(axis='y', style='sci',useOffset=False, scilimits=(-2,-2))
    x_ticks = np.linspace(0, np.pi, 4)  # 0, π/3, 2π/3, π
    axs[1, 1].set_xticks(x_ticks)
    axs[1, 1].set_xticklabels([r"$0$", r"$-\frac{\pi}{3}$", r"$-\frac{2\pi}{3}$", r"$-\pi$"])
    axs[1, 1].legend()

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig(f"{path}.pdf", bbox_inches="tight")
    plt.clf()

def load_problem(problem):
    pr = problem.NavierStokesProblem()
    return pr

def read_hdf(name):
    with fd.CheckpointFile(name+'.h5', 'r') as afile:
        mesh = afile.load_mesh()
        z = afile.load_function(mesh, "solution")
    return z

if __name__ == "__main__":
    # Dummy load, we take space directly from the solution. It is same for all the solutions.
    # If we had uploaded data, we would load all of them. Instead, we compute below directly.
    U = read_hdf('Umax=7.600000000000005e-01/solution-0')
    # Build spaces for traction (linear problem)
    W = U.function_space()
    Umax_list = [0.015] + [0.015*i for i in range(5,25,5)]

    drag_values_list = []
    lift_values_list = []
    Re_list = (np.array(Umax_list)*2.0/3.0*100).round(0).astype(int)
    dimless_const = 1.0 # We are outputing physicall pointwise drag and lift coefs instead of dimless coefs with /0.002
    for Umax in Umax_list:
        print(f"{Umax=}, Re = {Umax/3.*2.*100.}")
        z = compute_solution(W,Umax)
        (drag_coords,drag_values,lift_coords,lift_values) = compute_traction_bndryval(z)
        drag_values = (np.array(drag_values)*(-1)*dimless_const).tolist()
        lift_values = (np.array(lift_values)*(-1)*dimless_const).tolist()

        drag_values_list.append(drag_values)
        lift_values_list.append(lift_values)

    merged_drag_coords, merged_drag_values_list = merge_data(drag_coords, drag_values_list)
    merged_lift_coords, merged_lift_values_list = merge_data(lift_coords, lift_values_list)

    if not os.path.exists(f"traction_profiles/"):
        os.makedirs(f"traction_profiles/")

    plot_profiles_multiple(merged_drag_coords,merged_drag_values_list,Re_list,r"drag profile [N/m]","traction_profiles/pointwise_drag")
    plot_profiles_multiple(merged_lift_coords,merged_lift_values_list,Re_list,r"lift profile [N/m]","traction_profiles/pointwise_lift")