import gmsh
import matplotlib.pyplot as plt

# 1. Initialize Gmsh and load the geometry/mesh
gmsh.initialize()
gmsh.open("cavity.geo")

gmsh.option.setNumber("Mesh.Algorithm", 2)

# Generate 2D mesh if not already pre-computed
gmsh.model.mesh.generate(2)

# 2. Extract node coordinates
node_tags, coords, _ = gmsh.model.mesh.getNodes()
# Reshape flattened array (x1, y1, z1, x2, y2, z2, ...) into (N, 3)
node_coords = coords.reshape(-1, 3)

# Map node tags to 0-based array indices for triangulation
tag_to_index = {tag: i for i, tag in enumerate(node_tags)}

# 3. Extract 2D triangle element connectivity
element_types, element_tags, node_connectivity = gmsh.model.mesh.getElements(
    dim=2
)

triangles = []
for elem_type, nodes in zip(element_types, node_connectivity):
    # Element type 2 corresponds to 3-node triangles in Gmsh
    if elem_type == 2:
        # Reshape into triangles (each row has 3 node tags)
        elem_nodes = nodes.reshape(-1, 3)
        for tri_tags in elem_nodes:
            triangles.append([tag_to_index[tag] for tag in tri_tags])

gmsh.finalize()

# 4. Plot using Matplotlib
fig, ax = plt.subplots(figsize=(6, 5))

# Plot mesh edges
ax.triplot(
    node_coords[:, 0],
    node_coords[:, 1],
    triangles,
    color="black",
    linewidth=1,
)

# Plot nodes
ax.plot(node_coords[:, 0], node_coords[:, 1], "ko", markersize=5)

# Style & Save Transparent PNG
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

plt.savefig(
    "advancing_front.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
    transparent=True,
)
plt.close()
