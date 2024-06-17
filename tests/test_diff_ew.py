from ngsolve import CF, Mesh, x, y, sqrt, Integrate
from netgen.geom2d import unit_square
import ngsolve_addon_template_withdiff as addon

mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
func = CF((1, y,
           y, x)).Reshape((2, 2))
ews = addon.EigH(func)


def test_ew_diff():

    # hand-computed derivatives of exact eigenvalues wrt x:
    dews0 = 0.5 + 0.25 * (x-1) / sqrt(y*y + (1-x)**2/4)
    dews1 = 0.5 - 0.25 * (x-1) / sqrt(y*y + (1-x)**2/4)

    # add-on computed derivatives of exact eigenvalues wrt x:
    dews = ews.Diff(x)

    # differences:
    diff0 = Integrate((dews[0] - dews0)**2, mesh)
    diff1 = Integrate((dews[1] - dews1)**2, mesh)

    print('diffs = ', diff0, diff1)
    assert diff0 < 1e-30 and diff1 < 1e-30, \
        'Hand computed derivatives differ from add-on computed ones'


test_ew_diff()
