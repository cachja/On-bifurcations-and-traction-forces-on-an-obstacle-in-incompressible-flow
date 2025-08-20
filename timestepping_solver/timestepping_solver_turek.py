# Firedrake with PETSc required
import firedrake as fd
import os


def negpart(s):
    return fd.conditional(fd.gt(s, 0.0), 0.0, 1.0)*s


def read_hdf(name):
    with fd.CheckpointFile(name+'.h5', 'r') as afile:
        mesh = afile.load_mesh()
        z = afile.load_function(mesh, "solution")
    return z


def generate_ngmesh(maxh=0.02, circle_maxh=0.005):
    from netgen.geom2d import SplineGeometry
    import netgen

    if fd.COMM_WORLD.rank == 0:
        geo = SplineGeometry()
        geo.AddRectangle((0, 0), (2.2, 0.41), bcs=(2, 3, 2, 1))
        geo.AddCircle((0.2, 0.2), r=0.05, leftdomain=0,
                      rightdomain=1, bc=5, maxh=circle_maxh)
        ngmesh = geo.GenerateMesh(maxh=maxh)

        # Refine manually (near the outflow important for higher Re when vortices meets the outflow)
        for element in ngmesh.Elements2D():
            nodes = [ngmesh[ep] for ep in element.points]
            # 1. Refine near the right wall (outflow) - x > 2.0
            if all(n.p[0] > 2.0 for n in nodes):
                element.refine = True

            # 2. Refine around the inflow corners (0, 0) and (0, 41)
            elif any((n.p[0] < 0.01 and n.p[1] < 0.01) or (n.p[0] < 0.01 and n.p[1] > 0.4) for n in nodes):
                element.refine = True

            # 3. Refine around the circle (inside the box (0.05, 0.05) x (0.5, 0.35))
            elif any(0.05 < n.p[0] < 0.5 and 0.05 < n.p[1] < 0.35 for n in nodes):
                element.refine = True

            # If none of the conditions match, don't refine
            else:
                element.refine = False
        ngmesh.Refine(adaptive=True)
        # # Full red-green refinement
        # ngmesh.Refine()
        # # Full barycentric refinement
        # ngmesh.SplitAlfeld()
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(2)
    # Possibility of having curved element boundary
    # return fd.Mesh(fd.Mesh(ngmesh,comm=fd.COMM_WORLD).curve_field(3))
    return fd.Mesh(ngmesh, comm=fd.COMM_WORLD)


def compute_traction(U, U0, dt, V, a, b):
    v, p = U.subfunctions
    (v0, p0) = U0.subfunctions
    g = fd.TrialFunction(V)
    g_ = fd.TestFunction(V)
    fd.info(f"{V.dim()=}")

    # set BC
    # zero traction on bndries not touching bndry of interest
    bc_list = [1, 2, 3]
    bc_num = 5  # circle bndry - here we solve PS
    bcs_g = [fd.DirichletBC(V, fd.Constant((0, 0)), i) for i in bc_list]

    # Define the bilinear form for disrete variational traction problem
    a_g = fd.inner(g, g_) * fd.ds(bc_num)
    L_g = fd.Constant(1.0/dt)*fd.inner((v-v0), g_)*fd.dx + a(v, g_) - b(p, g_)

    # Solve linear system
    gg = fd.Function(V)
    A = fd.assemble(a_g, bcs=bcs_g)
    b = fd.assemble(L_g)

    # Dirty ident_zeros trick
    diagonal = A.petscmat.getDiagonal()
    vals = diagonal.array
    vals[vals == 0] = 1
    A.petscmat.setDiagonal(diagonal, fd.PETSc.InsertMode.INSERT_VALUES)

    fd.info("Solving Poincare steklov using sparse LU")
    fd.solve(A, gg, b, solver_parameters={"ksp_type": "preonly",
                                          "pc_type": "lu",
                                          "pc_factor_mat_solver_type": "mumps"})
    fd.info("PS solved")
    return gg


def continue_timestepping(wstart, mesh, Umax, branch):
    comm = fd.COMM_WORLD
    rank = comm.Get_rank()

    info = fd.PETSc.Sys.Print
    fd.logging.set_log_level(fd.DEBUG)  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    fd.info(
        f"Running Turek-Schaffer Benchmark from DefCon Steady state at Re = {Umax/3.*2.*100.}")

    VV = fd.VectorFunctionSpace(mesh, "CG", 2)
    Q = fd.FunctionSpace(mesh, "CG", 1)
    W = fd.MixedFunctionSpace([VV, Q])
    V = fd.VectorFunctionSpace(mesh, "CG", 1)
    info(f"#DoF {W.dim()}")

    x = fd.SpatialCoordinate(mesh)
    n = fd.FacetNormal(mesh)
    I = fd.Identity(mesh.geometric_dimension())
    Umax_const = fd.Constant(Umax)
    inflow_profile = fd.as_vector(
        [4.0*Umax_const*x[1]*(0.41-x[1])/0.41/0.41, 0])
    bc_inlet = fd.DirichletBC(W.sub(0), inflow_profile, 1)
    bc_cylinder = fd.DirichletBC(W.sub(0), (0, 0), 5)
    bc_walls = fd.DirichletBC(W.sub(0), (0, 0), 2)
    # traction free condition on outlet
    bcs = [bc_inlet, bc_cylinder, bc_walls]

    # Define unknown and test function(s)
    (v_, p_) = fd.TestFunctions(W)

    # current unknown time step
    w = fd.Function(W)
    if wstart != None:
        w.interpolate(wstart)
    (v, p) = fd.split(w)

    # previous known time step
    w0 = fd.Function(W)
    if wstart != None:
        w0.interpolate(wstart)
    (v0, p0) = fd.split(w0)

    dt = 0.005
    # dt = 1.0/(150*Umax) --- fit based on expected shedding frequency a CFL condition
    t_end = 10.0
    n_steps = t_end / dt
    theta = fd.Constant(0.5)   # Crank-Nicholson timestepping
    nu = fd.Constant(0.001)

    def a(v, u):
        D = fd.sym(fd.grad(v))
        # EMAC formulation
        return (fd.inner(2.0*D*v + fd.div(v)*v, u) + fd.inner(2.0*nu*D, fd.grad(u)))*fd.dx
        # inner(2*D*v + div(v)*v, u)

    def b(q, v):
        return fd.inner(fd.div(v), q)*fd.dx

    # Coordinates
    x = fd.SpatialCoordinate(mesh)

    # variational form without time derivative in current time
    F1 = a(v, v_) - b(p_, v) - b(p, v_)

    # variational forms without time derivative in previous time
    F0 = a(v0, v_) - b(p_, v) - b(p, v_)

    # combine variational forms with time derivative
    #
    #  dw/dt + F(w,t) = 0 is approximated as
    #  (w-w0)/dt + theta*F(w,t) + (1-theta)*F(w0,t0) = 0
    #

    dt_const = fd.Constant(1.0/dt)
    F = dt_const*fd.inner((v-v0), v_)*fd.dx + theta*F1 + (1.0-theta)*F0

    # # Backflow penalisation
    # directional_outflow_switch = Constant(0.0)
    # F += directional_outflow_switch*(-0.5*negpart(inner(v,n))*inner(v,v_))*ds(3) #3 is outflow

    J = fd.derivative(F, w)

    problem = fd.NonlinearVariationalProblem(F, w, bcs=bcs, J=J)
    lu = {"mat_type": "aij",
          "snes_type": "newtonls",
          "snes_monitor": None,
          "snes_converged_reason": None,
          "snes_max_it": 30,
          "snes_rtol": 1e-15,
          "snes_atol": 5e-16,
          "snes_linesearch_type": "basic",
          # Reuse Jacobian --- good tactics but not robust implementation w.r.t Re when initializing timestepping from zero
          "snes_lag_jacobian": 20,
          "snes_lag_jacobian_persists": True,
          "snes_lag_preconditioner": 20,  # reuse preconditioner too
          "snes_lag_preconditioner_persists": True,
          # Reuse Jacobian end
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps",
          "mat_mumps_icntl_14": 500,  # work array, multiple of estimate to allocate
          "mat_mumps_icntl_24": 1,  # detect null pivots
          "mat_mumps_cntl_1": 1.0}  # pivoting threshold, this solves to machine precision
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=lu)

    t = 0

    if not os.path.exists(f"output/Umax={Umax}/"):
        os.makedirs(f"output/Umax={Umax}/")
    f = open(f"output/Umax={Umax}/quantities_branch{branch}.txt", "w")

    step_counter = 0
    filevp = fd.VTKFile(f"output/Umax={Umax}/vpt_branch{branch}.pvd")
    while t < t_end:
        progress = 100.0 * t / t_end
        info(
            f"Step {step_counter}/{n_steps}, t = {t:.3f}, {progress:.1f}% complete")

        # update time-dependent parameters
        t += dt
        # Compute
        solver.solve()

        # # Report drag and lift
        dimless = fd.Constant(-2.0/(0.2*0.2*0.1))
        g = compute_traction(w, w0, dt, V, a, b)
        D1 = fd.assemble(dimless*g[0]*fd.ds(5))
        L1 = fd.assemble(dimless*g[1]*fd.ds(5))
        info(f"Discrete variation traction method: drag = {D1}, lift = {L1}")

        v, p = w.subfunctions

        def force(v, p, normal): return (
            nu*(fd.grad(v)+fd.grad(v).T)-p*I)*normal
        D2 = fd.assemble(dimless*force(v, p, n)[0]*fd.ds(5))
        L2 = fd.assemble(dimless*force(v, p, n)[1]*fd.ds(5))
        info(f"Direct traction evaluation: drag = {D2}, lift = {L2}")
        if rank == 0:
            f.write(f"t = {t}\t drag = {D1}\t lift={L1}\n")
            f.flush()

        # Save snapshow for Paraview every 100 steps
        if step_counter % 100 == 0:
            v, p = w.subfunctions
            v.rename("velocity")
            p.rename("pressure")
            g.rename("traction")
            filevp.write(v, p, g,  time=t)

        # Move w to w0 (only if solve() succeeds)
        w0.assign(w)

        step_counter += 1

        # Directional outflow
        # if condition in t e.g.:
        #   info("directional outflow ON")
        #   directional_outflow_switch.assign(1.0)

    f.close()


if __name__ == "__main__":
    Umax = 0.76
    # Either start from the steady state: branch = 0 for baseline steady branch
    z = read_hdf(f"Umax=7.600000000000005e-01/solution-0")
    # And use original mesh from the solution
    mesh = z.function_space().mesh()
    # or use a new netgen one - e.g. with higher order bndry
    # mesh = generate_ngmesh()

    # Or start from zero IC: branch = 6
    # z = None

    # Call the procedure
    branch = 0
    continue_timestepping(z, mesh, Umax, branch)
