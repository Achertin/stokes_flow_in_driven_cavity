import ufl
from dolfinx import (
    fem,
)
from dolfinx.io.gmsh import MeshData
from petsc4py import PETSc
from basix.ufl import (
            element,
            enriched_element,
            blocked_element
)

def setup_problem(
    mesh_data: MeshData, 
    v_element: str = "Lagrange", 
    q_element: str ="Lagrange", 
    v_degree: int = 2, 
    q_degree: int = 1 
):
    mesh = mesh_data.mesh
    bubble_element = enriched_element(
        [
            element("Lagrange", "triangle", 1,),
            element("Bubble", "triangle", 3)
        ]
    )
    P1_bubble = blocked_element(bubble_element, shape=(2,))
    V = fem.functionspace(mesh, (v_element, v_degree, (2,)))
    Q = fem.functionspace(mesh, (q_element, q_degree))
    #V = fem.functionspace(mesh, P1_bubble)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)

    f0 = fem.Constant(mesh, PETSc.ScalarType((0, 0)))
    f1 = fem.Constant(mesh, PETSc.ScalarType(0))

    a00 = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    a01 = -p * ufl.div(v) * ufl.dx
    a10 = -q * ufl.div(u) * ufl.dx
    a11 = None

    L0 = ufl.inner(f0, v) * ufl.dx
    L1 = f1 * q * ufl.dx

    bcs = setup_boundary_conditions(mesh_data, V)
    a = [[a00, a01], [a10, a11]]
    L = [L0, L1]
    print("Info    : Done setting up problem")
    return a, L, bcs

def setup_boundary_conditions(mesh_data: MeshData, functionspace: fem.FunctionSpace):
    mesh = mesh_data.mesh
    u_wall = fem.Constant(mesh, PETSc.ScalarType((0, 0)))
    u_top = fem.Constant(mesh, PETSc.ScalarType((1, 0))) 

    tag_wall = mesh_data.physical_groups["wall"].tag 
    tag_top = mesh_data.physical_groups["top"].tag

    dim = 1
    assert mesh_data.facet_tags is not None

    entities_wall = mesh_data.facet_tags.find(tag_wall)
    entities_top = mesh_data.facet_tags.find(tag_top)

    dofs_wall = fem.locate_dofs_topological(functionspace, dim, entities_wall)
    dofs_top = fem.locate_dofs_topological(functionspace, dim, entities_top)

    bc_wall = fem.dirichletbc(u_wall, dofs_wall, functionspace)
    bc_top = fem.dirichletbc(u_top, dofs_top, functionspace)
    return [bc_wall, bc_top]
