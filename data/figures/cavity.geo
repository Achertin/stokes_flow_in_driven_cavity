mesh_size = 1/6;

// Outer square points
Point(1) = {0, 0, 0, mesh_size};
Point(2) = {1, 0, 0, mesh_size};
Point(3) = {1, 1, 0, mesh_size};
Point(4) = {0, 1, 0, mesh_size};

// Outer square lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Outer boundary loop
Curve Loop(1) = {1, 2, 3, 4};

// --- CIRCULAR CUTOUT GEOMETRY ---
cx = 0.5; cy = 0.5; r = 0.2;  // Center (0.5, 0.5) and radius 0.2

Point(5) = {cx, cy, 0, mesh_size};       // Center point
Point(6) = {cx + r, cy, 0, mesh_size};   // Right
Point(7) = {cx, cy + r, 0, mesh_size};   // Top
Point(8) = {cx - r, cy, 0, mesh_size};   // Left
Point(9) = {cx, cy - r, 0, mesh_size};   // Bottom

// Define circle arcs using center point 5
Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

// Inner hole boundary loop
Curve Loop(2) = {5, 6, 7, 8};

// --- DEFINING SURFACE WITH HOLE ---
// Listing loop 2 inside Plane Surface subtracts the hole from loop 1
Plane Surface(1) = {1, 2};

// Physical Groups
Physical Curve("wall") = {1, 2, 4};
Physical Curve("top") = {3};
Physical Curve("hole") = {5, 6, 7, 8};
Physical Surface("domain") = {1};
