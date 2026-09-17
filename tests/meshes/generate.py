"""Generate the cantilever-plate test meshes (deterministic, no git-LFS needed).

    python tests/meshes/generate.py

Writes ``plate_<W>x<L>_{quad,tri}_<ny>x<nx>.xdmf`` (+ .h5). Node ordering is
row-major in (x fastest), which both dolfinx 0.5.1 and 0.11 read identically. The
``tri`` mesh (each quad split into two triangles) is for the CG2CR1 element.
"""
import pathlib

import numpy as np
import meshio

HERE = pathlib.Path(__file__).parent


def plate(width=2.0, length=10.0, ny=4, nx=20):
    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, width, ny + 1)
    points = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=float)
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n3 = n0 + (nx + 1)
            n2 = n3 + 1
            cells.append([n0, n1, n2, n3])
    return meshio.Mesh(points, [("quad", np.array(cells, dtype=np.int64))])


def plate_tri(width=2.0, length=10.0, ny=4, nx=20):
    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, width, ny + 1)
    points = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=float)
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n3 = n0 + (nx + 1)
            n2 = n3 + 1
            cells += [[n0, n1, n2], [n0, n2, n3]]
    return meshio.Mesh(points, [("triangle", np.array(cells, dtype=np.int64))])


def main():
    for ny, nx in [(4, 20), (8, 40)]:
        m = plate(ny=ny, nx=nx)
        path = HERE / f"plate_2x10_quad_{ny}x{nx}.xdmf"
        meshio.write(str(path), m)
        print(f"wrote {path}  ({len(m.points)} nodes, {len(m.cells[0].data)} quads)")
    for ny, nx in [(4, 20)]:
        m = plate_tri(ny=ny, nx=nx)
        path = HERE / f"plate_2x10_tri_{ny}x{nx}.xdmf"
        meshio.write(str(path), m)
        print(f"wrote {path}  ({len(m.points)} nodes, {len(m.cells[0].data)} tris)")


if __name__ == "__main__":
    main()
