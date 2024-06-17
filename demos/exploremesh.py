# Exploring the mesh in C++
from ngsolve import *
import ngsolve_addon_template_withdiff as addon

mesh = Mesh(unit_cube.GenerateMesh(maxh=1))

# explore mesh in C++, output data using cout:
addon.ExploreMesh(mesh)


# return results to Python:
for v in range(mesh.nv):
    print (v, addon.Vertex2Element(mesh,v))
