from src.mesh_setup import generate_mesh
from src.problem_builder import setup_problem
from mpi4py import MPI
from dolfinx.io import VTXWriter
from dolfinx.fem.petsc import LinearProblem
from dolfinx import fem

MESH_SIZE = 1/16
NAME = "cavity_P1-P1_8"
mesh_data = generate_mesh(MESH_SIZE)
a, L, bcs = setup_problem(mesh_data, "Lagrange", "Lagrange", 1, 1)

# setting up and solving the system
problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    petsc_options_prefix="cavity_",
    kind="nest",
    petsc_options= {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1,   # detect the null pivot (pressure constant)
        "mat_mumps_icntl_25": 0,
        "ksp_error_if_not_converged": True,
    } 
)
u_sol, p_sol = problem.solve()
assert u_sol and p_sol is not None
print("velocity DOF: ", u_sol.x.index_map.size_global * u_sol.x.block_size)
print("pressure DOF: ", p_sol.x.index_map.size_global * p_sol.x.block_size)

print("Info    : Done solving system")

# defning common function space
V0 = fem.functionspace(mesh_data.mesh, ("Lagrange", 1, (2,)))
V1 = fem.functionspace(mesh_data.mesh, ("Lagrange", 1))
u_out = fem.Function(V0, name="velocity")
p_out = fem.Function(V1, name="pressure")
u_out.interpolate(u_sol)
p_out.interpolate(p_sol)

with VTXWriter (
    MPI.COMM_WORLD, f"data/results/{NAME}.bp", [u_out, p_out], engine="BP4"
) as vtx:
    vtx.write(0.0)
print("Info    : Done writing data")
