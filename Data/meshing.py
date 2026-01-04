import gmsh

STEP_FILE = "data/Geometry.STEP"
OUT_MSH = "data/GeometryAuto.msh"
MESH_SIZE = 0.1  # start coarse, refine later

gmsh.initialize()
gmsh.model.add("beam")

gmsh.model.occ.importShapes(STEP_FILE)
gmsh.model.occ.synchronize()

gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

volumes = gmsh.model.getEntities(dim=3)
if not volumes:
    raise RuntimeError("No volume found in STEP file")

gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], 1)
gmsh.model.setPhysicalName(3, 1, "beam")

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", MESH_SIZE)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", MESH_SIZE)

gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

gmsh.option.setNumber("Mesh.Algorithm", 6)     # 2D
gmsh.option.setNumber("Mesh.Algorithm3D", 4)   # 3D

gmsh.option.setNumber("Mesh.Optimize", 0)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)

gmsh.option.setNumber("Mesh.SaveAll", 0)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

gmsh.model.mesh.generate(2)
gmsh.write(OUT_MSH)

gmsh.finalize()

print("Mesh generated:", OUT_MSH)
