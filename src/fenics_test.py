# monolithic implementation of fenics to calculate stokes problem
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from basix.ufl import element
from dolfinx import fem
import ufl
from petsc4py import PETSc
import ufl
from dolfinx.fem.petsc import LinearProblem
from pathlib import Path
from dolfinx.io import VTXWriter

home_path = Path().home()
mesh_path = home_path / Path("projects/fenicsx/data/meshes/cavity.msh")
results_path = home_path / Path("projects/fenicsx/data/results")
results_path.mkdir(exist_ok=True, parents=True)

# reading mesh form a .msh file
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
mesh_data = gmshio.read_from_msh(mesh_path, mesh_comm, gmsh_model_rank, gdim=2)
assert mesh_data.cell_tags is not None
domain = mesh_data.mesh

# defining function spaces for velocity and pressure
v_cg2 = element("Lagrange", domain.basix_cell(), 2, shape=(domain.geometry.dim,))
v_cg1 = element("Lagrange", domain.basix_cell(), 1)
V = fem.functionspace(domain, v_cg2)
Q = fem.functionspace(domain, v_cg1)

# test and trial functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
p = ufl.TrialFunction(Q)
q = ufl.TestFunction(Q)

f0 = fem.Constant(domain, PETSc.ScalarType((0, 0)))
f1 = fem.Constant(domain, PETSc.ScalarType(0))

# defining forms
a00 = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
a01 = -p * ufl.div(v) * ufl.dx
a10 = q * ufl.div(u) * ufl.dx
a11 = None

L0 = ufl.inner(f0, v) * ufl.dx
L1 = f1 * q * ufl.dx

# defining boundary conditions
u_wall = fem.Constant(domain, PETSc.ScalarType((0, 0)))
u_top = fem.Constant(domain, PETSc.ScalarType((1, 0)))

domain_wall_tag = mesh_data.physical_groups["wall"].tag
domain_top_tag = mesh_data.physical_groups["top"].tag
domain_wall_dim = mesh_data.physical_groups["wall"].dim
domain_top_dim = mesh_data.physical_groups["top"].dim

domain_wall_entities = mesh_data.facet_tags.find(domain_wall_tag)
domain_top_entities = mesh_data.facet_tags.find(domain_top_tag)

domain_wall_dofs = fem.locate_dofs_topological(V, domain_wall_dim, domain_wall_entities)
domain_top_dofs = fem.locate_dofs_topological(V, domain_top_dim, domain_top_entities)

bc_wall = fem.dirichletbc(u_wall, domain_wall_dofs, V)
bc_top = fem.dirichletbc(u_top, domain_top_dofs, V)

# solving the algebraic system
problem = LinearProblem(
    [[a00, a01], [a10, a11]],
    [L0, L1],
    bcs=[bc_top, bc_wall],
    kind="nest",
    petsc_options_prefix="nest_linear_problem"
)
uh = problem.solve()
uh[0].name = 'velocity'
uh[1].name = 'pressure'

# saving the solution

with VTXWriter (
    MPI.COMM_WORLD, results_path / "cavity_u.bp", uh[0], engine="BP4"
) as vtx:
    vtx.write(0.0)
with VTXWriter (
    MPI.COMM_WORLD, results_path / "cavity_p.bp", uh[1], engine="BP4"
) as vtx:
    vtx.write(0.0)
