import numpy as np
import ufl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from src.mesh_setup import generate_mesh
from src.problem_builder import setup_problem

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.size": 22,
})


def solve_cavity(mesh_data, v_deg, q_deg, name):
    a, L, bcs = setup_problem(mesh_data, "Lagrange", "Lagrange", v_deg, q_deg)
    problem = LinearProblem(
        a, L, bcs=bcs, kind="mpi",
        petsc_options_prefix=f"{name}_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1,
            "mat_mumps_icntl_25": 0,
            "ksp_error_if_not_converged": True,
        },
    )
    return problem.solve()


def p1_triangulation(p):
    """Interpolate a scalar field into P1 and return a matplotlib triangulation plus
    zero-mean nodal values (serial only)."""
    msh = p.function_space.mesh
    V1 = fem.functionspace(msh, ("Lagrange", 1))
    p1 = fem.Function(V1)
    p1.interpolate(p)

    xy = V1.tabulate_dof_coordinates()[:, :2]
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1], V1.dofmap.list)

    # the pressure is only fixed up to a constant: remove its mean
    area = fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ufl.dx))
    mean = fem.assemble_scalar(fem.form(p1 * ufl.dx)) / area
    return tri, p1.x.array.real - mean


mesh_data = generate_mesh(1/16)
cases = {
    r"P1--P1": (1, 1),
    r"P2--P1 (Taylor--Hood)": (2, 1),
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
for ax, (label, (v_deg, q_deg)) in zip(axes, cases.items()):
    u_sol, p_sol = solve_cavity(mesh_data, v_deg, q_deg, f"P{v_deg}P{q_deg}")
    tri, vals = p1_triangulation(p_sol)

    # the corner singularities dominate the range: clip the colour scale
    # to the 2-98 % percentiles so the interior pattern is visible
    lim = np.max(np.abs(np.percentile(vals, [2, 98])))
    tpc = ax.tripcolor(tri, vals, shading="gouraud", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.triplot(tri, color="k", lw=0.3, alpha=0.4)        # mesh edges
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(label)
    fig.colorbar(tpc, ax=ax, shrink=0.8, label=r"$p - \bar p$")

fig.suptitle(r"\textbf{Pressure in the lid-driven cavity}")
plt.savefig("/home/lukasl/Documents/vienna_uni/computational_analysis_seminar/comp_math_presentation/images/pressure_comparison.png",
            dpi=300, bbox_inches="tight", transparent=True)
plt.show()
