import gmsh
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

def generate_mesh(mesh_size=0.1, model_name="cavity"):
    gmsh.initialize()
    mesh_size = mesh_size
    gmsh.model.add(f"{model_name}")

    # generating corner points of the domain
    gmsh.model.occ.addPoint(0, 0, 0, mesh_size, 1)
    gmsh.model.occ.addPoint(1, 0, 0, mesh_size, 2)
    gmsh.model.occ.addPoint(1, 1, 0, mesh_size, 3)
    gmsh.model.occ.addPoint(0, 1, 0, mesh_size, 4)

    # attaching points with lines
    gmsh.model.occ.addLine(1, 2, 1)
    gmsh.model.occ.addLine(2, 3, 2)
    gmsh.model.occ.addLine(3, 4, 3)
    gmsh.model.occ.addLine(4, 1, 4)

    # creating continous loop from lines
    gmsh.model.occ.addCurveLoop([1, 2, 3, 4], 1)

    # defining surface from loop
    gmsh.model.occ.addPlaneSurface([1], 1)

    # generating mesh
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(1, [1, 2, 4], name="wall")
    gmsh.model.addPhysicalGroup(1, [3], name="top")
    gmsh.model.addPhysicalGroup(2, [1], name="surface")

    gmsh.model.mesh.generate(2)

    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=2)
    assert mesh_data.facet_tags is not None
    gmsh.finalize()
    print("Info    : Done generating mesh")
    return mesh_data

if __name__ == "__main__":
    generate_mesh(mesh_size=1e-2, model_name="cavity")
