"""Pressure, velocity and corner-eddy plots for the lid-driven cavity.

Solves the Stokes cavity problem with P1-P1 and P2-P1 (Taylor-Hood) elements
and makes three figures:
    - pressure, both element pairs side by side (spurious modes of P1-P1)
    - velocity streamlines, both element pairs side by side
    - streamfunction contours zoomed into a bottom corner (Moffatt eddy),
      for Taylor-Hood

Run in serial:  python plot_cavity.py
"""

import numpy as np
import ufl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc

from src.mesh_setup import generate_mesh
from src.problem_builder import setup_problem
from src.plot_style import apply_style, save

MESH_SIZE = 1 / 16
GRID_POINTS = 200          # resolution of the regular grid used for streamlines

CASES = {
    r"P1--P1": (1, 1),
    r"P2--P1 (Taylor--Hood)": (2, 1),
}

apply_style()


def solve_cavity(mesh_data, v_deg, q_deg, name):
    """Solve the cavity problem with a direct solver; returns (u, p)."""
    a, L, bcs = setup_problem(mesh_data, "Lagrange", "Lagrange", v_deg, q_deg)
    problem = LinearProblem(
        a, L, bcs=bcs, kind="mpi",
        petsc_options_prefix=f"{name}_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1,   # detect the null pivot (pressure constant)
            "mat_mumps_icntl_25": 0,
            "ksp_error_if_not_converged": True,
        },
    )
    return problem.solve()


def p1_triangulation(f, remove_mean=False):
    """Interpolate a scalar field into P1; return a matplotlib triangulation
    and the nodal values (optionally with the mean removed)."""
    msh = f.function_space.mesh
    V1 = fem.functionspace(msh, ("Lagrange", 1))
    f1 = fem.Function(V1)
    f1.interpolate(f)

    xy = V1.tabulate_dof_coordinates()[:, :2]
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1], V1.dofmap.list)
    vals = f1.x.array.real.copy()

    if remove_mean:
        area = fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ufl.dx))
        vals -= fem.assemble_scalar(fem.form(f1 * ufl.dx)) / area
    return tri, vals


def velocity_on_grid(u, n=GRID_POINTS):
    """Interpolate a vector field into P1 and resample it on a regular n x n grid."""
    msh = u.function_space.mesh
    V1 = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    u1 = fem.Function(V1)
    u1.interpolate(u)

    xy = V1.tabulate_dof_coordinates()[:, :2]          # one point per vertex
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1], V1.dofmap.list)
    vals = u1.x.array.real.reshape(-1, 2)              # (ux, uy) per vertex

    X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    Ux = mtri.LinearTriInterpolator(tri, vals[:, 0])(X, Y)
    Uy = mtri.LinearTriInterpolator(tri, vals[:, 1])(X, Y)
    return X, Y, np.ma.filled(Ux, 0.0), np.ma.filled(Uy, 0.0)


def streamfunction(u):
    """Solve -Laplace(psi) = curl(u) with psi = 0 on the boundary (P2)."""
    msh = u.function_space.mesh
    W = fem.functionspace(msh, ("Lagrange", 2))
    psi, phi = ufl.TrialFunction(W), ufl.TestFunction(W)
    a = ufl.inner(ufl.grad(psi), ufl.grad(phi)) * ufl.dx
    L = (u[1].dx(0) - u[0].dx(1)) * phi * ufl.dx

    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    dofs = fem.locate_dofs_topological(W, fdim, dmesh.exterior_facet_indices(msh.topology))
    bc = fem.dirichletbc(PETSc.ScalarType(0), dofs, W)

    problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="psi_",
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                           "pc_factor_mat_solver_type": "mumps"})
    return problem.solve()


def format_field_axes(ax):
    ax.grid(False)                       # no grid lines on top of fields
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")


def plot_pressure(solutions):
    fig, axes = plt.subplots(1, len(solutions), figsize=(18, 8), constrained_layout=True)
    for ax, (label, (u_sol, p_sol)) in zip(axes, solutions.items()):
        tri, vals = p1_triangulation(p_sol, remove_mean=True)
        # The corner singularities dominate the range: clip the colour scale
        # to the 2-98 % percentiles so the interior pattern is visible
        lim = np.max(np.abs(np.percentile(vals, [2, 98])))
        tpc = ax.tripcolor(tri, vals, shading="gouraud", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.triplot(tri, color="k", lw=0.4, alpha=0.4)   # mesh edges
        ax.set_title(rf"\textbf{{{label}}}")
        fig.colorbar(tpc, ax=ax, shrink=0.8, label=r"$p - \bar p$")
        format_field_axes(ax)
    fig.suptitle(r"\textbf{Lid-driven cavity: pressure}", fontsize=32)
    save(fig, "pressure_comparison.png")


def plot_streamlines(solutions):
    fig, axes = plt.subplots(1, len(solutions), figsize=(18, 8), constrained_layout=True)
    for ax, (label, (u_sol, p_sol)) in zip(axes, solutions.items()):
        X, Y, Ux, Uy = velocity_on_grid(u_sol)
        speed = np.sqrt(Ux**2 + Uy**2)
        strm = ax.streamplot(X, Y, Ux, Uy, color=speed, cmap="viridis",
                             density=1.8, linewidth=1.8, arrowsize=1.8)
        ax.set_title(rf"\textbf{{{label}}}")
        fig.colorbar(strm.lines, ax=ax, shrink=0.8, label=r"$|u|$")
        format_field_axes(ax)
    fig.suptitle(r"\textbf{Lid-driven cavity: velocity streamlines}", fontsize=32)
    save(fig, "streamlines_comparison.png")


def plot_corner_eddy(u, corner=(0.0, 0.0), size=0.2):
    """Streamfunction contours zoomed into one bottom corner."""
    tri, vals = p1_triangulation(streamfunction(u))
    xy = np.column_stack((tri.x, tri.y))

    x0, y0 = corner
    in_box = (np.abs(xy[:, 0] - x0) < size) & (xy[:, 1] - y0 < size)
    eddy_max = vals[in_box].max()                       # strength of the corner eddy

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.tricontour(tri, vals, levels=np.linspace(vals.min(), 0, 30)[1:-1],
                  colors="tab:blue", linewidths=2.0)    # main vortex (psi < 0)
    if eddy_max > 0:
        ax.tricontour(tri, vals, levels=eddy_max * np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95]),
                      colors="tab:red", linewidths=2.0)  # corner eddy (psi > 0)
    ax.grid(False)
    ax.set_xlim((x0, x0 + size) if x0 == 0 else (x0 - size, x0))
    ax.set_ylim(y0, y0 + size)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(rf"\textbf{{Corner eddy}}, $\psi_{{\max}} = {eddy_max:.2e}$")
    save(fig, "corner_eddy.png")


def main():
    mesh_data = generate_mesh(MESH_SIZE)

    # Solve once per element pair, then plot from the stored solutions
    solutions = {
        label: solve_cavity(mesh_data, v_deg, q_deg, f"P{v_deg}P{q_deg}")
        for label, (v_deg, q_deg) in CASES.items()
    }

    plot_pressure(solutions)
    plot_streamlines(solutions)
    u_th, _ = solutions[r"P2--P1 (Taylor--Hood)"]
    plot_corner_eddy(u_th)
    plt.show()


if __name__ == "__main__":
    main()
