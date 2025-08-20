
# -*- coding: utf-8 -*-
import firedrake as fd
import defcon
import numpy as np
import matplotlib.pyplot as plt


class NavierStokesProblem(BifurcationProblem):
    def mesh(self, comm):
        # Markers: 1 = inlet, 3 = outlet, 2 = walls, cylinder = 5
        mesh = fd.Mesh('turek_mesh_36_03.msh', comm=comm)
        return mesh

    def function_space(self, mesh):
        self.V = fd.VectorFunctionSpace(mesh, "CG", 2)
        self.Q = fd.FunctionSpace(mesh, "CG", 1)
        Z = fd.MixedFunctionSpace([self.V, self.Q])
        return Z

    def parameters(self):
        Umax = fd.Constant(0.3)
        return [(Umax, "Umax", r"$U_{\rm max}$")]

    def residual(self, z, params, w):
        (u, p) = fd.split(z)
        (v, q) = fd.split(w)

        nu = 0.001
        mesh = z.function_space().mesh()

        F = (
            nu * fd.inner(fd.grad(u), fd.grad(v))*fd.dx
            + fd.inner(fd.grad(u)*u, v)*fd.dx
            - fd.div(v)*p*fd.dx
            + q*fd.div(u)*fd.dx
        )

        return F

    def boundary_conditions(self, Z, params):
        # Inlet BC
        Umax = params[0]
        x = fd.SpatialCoordinate(Z.mesh())
        inflow_profile = fd.as_vector([4.0*Umax*x[1]*(0.41-x[1])/0.41/0.41, 0])

        bc_inlet = fd.DirichletBC(Z.sub(0), inflow_profile, 1)
        bc_cylinder = fd.DirichletBC(Z.sub(0), (0, 0), 5)
        bc_walls = fd.DirichletBC(Z.sub(0), (0, 0), 2)
        bc_outlet = fd.DirichletBC(Z.sub(0).sub(1), 0, 3)
        bcs = [bc_inlet, bc_cylinder, bc_walls, bc_outlet]

        return bcs

    def functionals(self):
        def sqL2(z, params):
            (u, p) = fd.split(z)
            j = fd.assemble(fd.inner(u, u)*fd.dx)
            return j

        def diss(z, params):
            (u, p) = fd.split(z)
            D = 0.5*(fd.grad(u) + fd.grad(u).T)
            nu = 0.001
            j = fd.assemble(2*nu*fd.inner(D, D)*fd.dx)
            return j

        def lift(z, params):
            (u, p) = fd.split(z)
            I = fd.Identity(2)
            D = 0.5*(fd.grad(u) + fd.grad(u).T)
            nu = 0.001
            mesh = z.function_space().mesh()

            V = fd.VectorFunctionSpace(mesh, "CG", 2)
            Q = fd.FunctionSpace(mesh, "CG", 1)
            Z = fd.MixedFunctionSpace([V, Q])

            w = fd.TestFunction(Z)
            (v, q) = fd.split(w)
            F = fd.inner(-p*I+2*nu*D, fd.grad(v))*fd.dx - q * \
                fd.div(u)*fd.dx + fd.inner(fd.grad(u)*u, v)*fd.dx
            z_ = fd.Function(Z)
            fd.DirichletBC(Z.sub(0), (0.0, 1.0), 5).apply(z_.vector())
            lift = -fd.assemble(fd.action(F, z_))/0.002  # dimless lift coef

            return lift

        def drag(z, params):
            (u, p) = fd.split(z)
            I = fd.Identity(2)
            D = 0.5*(fd.grad(u) + fd.grad(u).T)
            nu = 0.001
            mesh = z.function_space().mesh()

            V = fd.VectorFunctionSpace(mesh, "CG", 2)
            Q = fd.FunctionSpace(mesh, "CG", 1)
            Z = fd.MixedFunctionSpace([V, Q])

            w = fd.TestFunction(Z)
            (v, q) = fd.split(w)
            F = fd.inner(-p*I+2*nu*D, fd.grad(v))*fd.dx - q * \
                fd.div(u)*fd.dx + fd.inner(fd.grad(u)*u, v)*fd.dx
            z_ = fd.Function(Z)
            fd.DirichletBC(Z.sub(0), (1.0, 0.0), 5).apply(z_.vector())
            drag = -fd.assemble(fd.action(F, z_))/0.002  # dimless drag coef

            return drag

        return [(sqL2, "sqL2", r"$\|u\|^2$"), (diss, "diss", r"$2\nu\|{\bf D}\|^2$"), (lift, "lift", r"Lift"), (drag, "drag", r"Drag")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, Z, params, n):
        return fd.Function(Z)

    def number_solutions(self, params):
        return float("inf")

    def save_pvd(self, rc, pvd, params):
        (u, p) = rc.split()
        u.rename("Velocity", "Velocity")
        p.rename("Pressure", "Pressure")
        pvd.write(u, p)

    def solver_parameters(self, params, task, **kwargs):
        params = {
            "mat_type": "aij",
            "snes_monitor": None,
            "snes_linesearch_type": "basic",
            "snes_max_it": 100,
            "snes_atol": 1.0e-9,
            "snes_rtol": 0.0,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"
        }
        return params


if __name__ == "__main__":
    dc = defcon.DeflatedContinuation(
        problem=NavierStokesProblem(), teamsize=1, verbose=True)
    dc.run(values={"Umax": list(np.arange(0.3, 15.0, 0.02))})

    dc.bifurcation_diagram("lift")
    plt.title(r"Bifurcation diagram for the Turek-Schafer benchmark")
    plt.savefig("bifurcation.pdf")
