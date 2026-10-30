mesh_size = 0.25;

// defining points
Point(1) = {0, 0, 0, mesh_size};
Point(2) = {1, 0, 0, mesh_size};
Point(3) = {1, 1, 0, mesh_size};
Point(4) = {0, 1, 0, mesh_size};

// connecting points with lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// defining a loop from lines
Curve Loop(1) = {1, 2, 3, 4};

// defining surface form loop
Plane Surface(1) = {1};

Physical Curve("wall") = {1, 2, 4};
Physical Curve("top") = {3};
Physical Surface("domain") = {1};

// Hide axes
General.Axes = 0;
General.SmallAxes = 0;

// Nodes (Points) styling
Mesh.Nodes = 1;
Mesh.NodeSize = 6;
Mesh.Color.Points = RGB(0, 0, 0);

// Lines styling
Mesh.Lines = 1;
Mesh.LineWidth = 1;
Mesh.SurfaceFaces = 0;
Mesh.Color.Lines = RGB(0, 0, 0);

// Transparent background export setting
Print.Background = 1;

// Mesh generation
Mesh 2;

// Export PNG
Print "mesh_transparent.png";
