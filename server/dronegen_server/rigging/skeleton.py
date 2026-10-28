"""A mesh -> its curve skeleton: a tree through the middle of every part, with the thickness at each point.

The mesh is filled into a voxel grid and the skeleton is built twice:

* thinning to one-voxel lines (``skimage.morphology.skeletonize``) gives clean lines on real models, but
  sometimes wears a whole part away (all four legs of a test horse went because of its thin tail);
* the layers of the distance *through the body* from its thickest point (a Reeb graph): cut into thin
  layers by that distance the body falls apart into slices, a disc across a leg, a ring across the body,
  and where a layer splits a branch begins. It never loses a part, but every bump is a branch too.

The thinned tree is the base; a long branch of the layers that it lost is grafted onto it. Short side twigs
(a bump on a thigh, a fold of a wing) are merged back into the branch they grow from, so what is left has
one branch per real part: legs, wings, a neck, a tail, ears, fingers of a wing. Every filled voxel belongs
to its nearest node: a rough split of the body into its parts.

Coordinates are the caller's (Blender object space: Z up); nothing here depends on where the front is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..errors import UserError

RESOLUTION = 128  # voxels along the longest side: a wing a few voxels thick is still solid
MAX_CELLS = 300_000  # filled voxels at most: the voxel graph of a bulky solid would take hundreds of megabytes
LAYER = 1.5  # voxels: the thickness of one layer of the distance wave (one slice of a limb)
SPUR_THIN = 1.2  # a twig of thinning shorter than this many radii of the node it grows from is noise
SPUR_LAYER = 2.0  # a branch of the layers shorter than this many half-widths of its slice is a bump
SPUR_MIN = 3.0  # voxels: twigs shorter than this always go
GRAFT_MIN = 4.0  # voxels: a branch the thinning lost is put back when it reaches at least this far


@dataclass
class Volume:
    """The filled voxel grid of a mesh. ``origin`` is the centre of voxel [0, 0, 0]."""

    grid: np.ndarray
    pitch: float
    origin: np.ndarray

    def world(self, ijk) -> np.ndarray:
        return self.origin + np.asarray(ijk, np.float64) * self.pitch

    def cells(self) -> np.ndarray:
        """World centres of the filled voxels."""
        return self.world(np.argwhere(self.grid))


@dataclass
class Tree:
    """An undirected tree: node positions (world), the thickness (distance to the surface) and neighbours."""

    points: np.ndarray
    radius: np.ndarray
    nbrs: list[list[int]]
    owner: np.ndarray = field(default=None, repr=False)  # node of every filled voxel (``Volume.cells`` order)

    def __len__(self) -> int:
        return len(self.points)

    def degree(self, i: int) -> int:
        return len(self.nbrs[i])

    def ends(self) -> list[int]:
        return [i for i in range(len(self)) if len(self.nbrs[i]) == 1]

    def rooted(self, root: int) -> tuple[np.ndarray, list[int]]:
        """(parent of every node, -1 at the root; nodes in breadth-first order from the root)."""
        parent = np.full(len(self), -1)
        order, seen = [root], {root}
        for i in order:
            for j in self.nbrs[i]:
                if j not in seen:
                    seen.add(j)
                    parent[j] = i
                    order.append(j)
        return parent, order

    def path(self, a: int, b: int) -> list[int]:
        parent, _ = self.rooted(a)
        out = [b]
        while out[-1] != a:
            if parent[out[-1]] < 0:
                raise ValueError("nodes are not connected")
            out.append(int(parent[out[-1]]))
        return out[::-1]

    def length(self, path: list[int]) -> float:
        p = self.points[path]
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(path) > 1 else 0.0

    def distances(self, root: int) -> np.ndarray:
        """Length along the tree from ``root`` to every node."""
        parent, order = self.rooted(root)
        d = np.zeros(len(self))
        for i in order[1:]:
            d[i] = d[parent[i]] + np.linalg.norm(self.points[i] - self.points[parent[i]])
        return d


def volume(vertices: np.ndarray, faces: np.ndarray, resolution: int = RESOLUTION) -> Volume:
    """Filled voxels of a mesh; a mesh with small holes is closed by one step of morphological closing first.
    A bulky solid (a yurt) gets a coarser grid, so that the graph of its voxels stays small."""
    import trimesh
    from scipy import ndimage

    mesh = trimesh.Trimesh(np.asarray(vertices, np.float64), np.asarray(faces, np.int64), process=False)
    extent = float(mesh.extents.max()) if len(mesh.vertices) else 0.0
    if extent <= 0 or len(mesh.faces) == 0:
        raise UserError("Пустая сетка: не из чего строить скелет")
    while True:
        pitch = extent / resolution
        vox = mesh.voxelized(pitch)
        shell = np.pad(np.asarray(vox.matrix, bool), 2)
        grid = ndimage.binary_fill_holes(shell)
        if grid.sum() < 1.5 * shell.sum():  # the surface had gaps, its inside did not fill: close them and try again
            grid = ndimage.binary_fill_holes(ndimage.binary_closing(shell, iterations=1))
        filled = int(grid.sum())
        if filled <= MAX_CELLS or resolution <= 48:
            break
        resolution = max(48, int(resolution * (MAX_CELLS / filled) ** (1 / 3)))
    origin = np.asarray(vox.transform)[:3, 3] - 2 * pitch
    return Volume(grid, pitch, origin)


def tree(vol: Volume) -> Tree:
    """The curve skeleton of a filled volume as a tree (positions and radii in world units)."""
    from scipy import ndimage
    from scipy.spatial import cKDTree

    cells = np.argwhere(vol.grid)
    edt = ndimage.distance_transform_edt(vol.grid)
    pts, rad, nbrs = _thinned(vol.grid, edt)
    pts, rad, nbrs = _graft(pts, rad, nbrs, *_layered(cells, edt[tuple(cells.T)], vol.grid.shape))
    owner = cKDTree(pts).query(cells)[1]  # every filled voxel belongs to its nearest node
    return Tree(points=vol.world(pts), radius=rad * vol.pitch, nbrs=[sorted(s) for s in nbrs],
                owner=np.asarray(owner, np.int64))


def _voxel_graph(cells: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edges (a, b, length) between 26-neighbouring voxels of ``cells``, every pair once."""
    index = -np.ones(tuple(np.asarray(shape) + 2), np.int64)
    index[tuple((cells + 1).T)] = np.arange(len(cells))
    rows, cols, lengths = [], [], []
    for o in np.argwhere(np.ones((3, 3, 3))) - 1:
        if tuple(o) <= (0, 0, 0):
            continue
        nb = index[tuple((cells + 1 + o).T)]
        ok = nb >= 0
        rows.append(np.nonzero(ok)[0])
        cols.append(nb[ok])
        lengths.append(np.full(int(ok.sum()), np.linalg.norm(o)))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(lengths)


def _components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy import sparse
    from scipy.sparse import csgraph

    graph = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)).tocsr()
    return csgraph.connected_components(graph, directed=False)[1]


def _spanning(n: int, a: np.ndarray, b: np.ndarray, length: np.ndarray) -> tuple[list[set[int]], np.ndarray]:
    """Neighbour sets of the minimum spanning tree of a graph, and which nodes are in its largest piece."""
    from scipy import sparse
    from scipy.sparse import csgraph

    piece = _components(n, a, b)
    alive = piece == np.bincount(piece).argmax()
    span = csgraph.minimum_spanning_tree(sparse.coo_matrix((length + 1e-6, (a, b)), shape=(n, n))).tocoo()
    nbrs = [set() for _ in range(n)]
    for i, j in zip(span.row, span.col):
        if alive[i]:
            nbrs[i].add(int(j))
            nbrs[j].add(int(i))
    return nbrs, alive


def _compact(pos: np.ndarray, radius: np.ndarray, nbrs: list[set[int]], alive: np.ndarray):
    keep = np.nonzero(alive)[0]
    new = -np.ones(len(pos), np.int64)
    new[keep] = np.arange(len(keep))
    return (np.asarray(pos, np.float64)[keep], np.asarray(radius, np.float64)[keep],
            [{int(new[j]) for j in nbrs[i]} for i in keep])


def _thinned(grid: np.ndarray, edt: np.ndarray):
    """Thinning to one-voxel lines (voxel units): (positions, radii, neighbours) of their spanning tree."""
    from skimage.morphology import skeletonize

    idx = np.argwhere(skeletonize(grid) > 0)
    if len(idx) == 0:  # thinner than a voxel everywhere: the middle of the whole thing
        idx = np.round(np.argwhere(grid).mean(axis=0, keepdims=True)).astype(int)
    radius = edt[tuple(idx.T)]
    a, b, step = _voxel_graph(idx, grid.shape)
    nbrs, alive = _spanning(len(idx), a, b, step)
    merged = np.arange(len(idx))
    _prune(idx.astype(np.float64), radius, nbrs, merged, SPUR_THIN)
    return _compact(idx, radius, nbrs, alive & (merged == np.arange(len(idx))))


def _layered(cells: np.ndarray, edt: np.ndarray, shape):
    """The Reeb graph of the distance through the body from its thickest point (voxel units): (positions, radii,
    neighbours, the root node)."""
    from scipy import sparse
    from scipy.sparse import csgraph

    n = len(cells)
    a, b, step = _voxel_graph(cells, shape)
    piece = _components(n, a, b)
    main = piece == np.bincount(piece).argmax()  # specks apart from the body are left out
    root = int(np.argmax(np.where(main, edt, -1.0)))
    dist = csgraph.dijkstra(sparse.coo_matrix((step, (a, b)), shape=(n, n)).tocsr(), directed=False, indices=root)
    layer = np.where(np.isfinite(dist), np.floor(dist / LAYER), -1).astype(np.int64)
    inside = np.nonzero(layer >= 0)[0]

    # slices: voxels of one layer that touch each other
    same = (layer[a] == layer[b]) & (layer[a] >= 0)
    slice_of = _components(n, a[same], b[same])
    ids, slice_of[inside] = np.unique(slice_of[inside], return_inverse=True)
    m = len(ids)
    counts = np.bincount(slice_of[inside], minlength=m).astype(np.float64)
    centre = np.stack([np.bincount(slice_of[inside], cells[inside, k], minlength=m) for k in range(3)], 1)
    centre /= counts[:, None]
    # the thickness of a slice: the distance to the surface at its voxel nearest to the centre
    d2 = ((cells[inside] - centre[slice_of[inside]]) ** 2).sum(1)
    order = np.lexsort((d2, slice_of[inside]))
    first = np.unique(slice_of[inside][order], return_index=True)[1]
    radius = edt[inside[order[first]]]

    # slices of neighbouring layers that touch are joined; loops (an arm resting on a hip) are cut by a spanning tree
    cross = (np.abs(layer[a] - layer[b]) == 1) & (layer[a] >= 0) & (layer[b] >= 0)
    pairs = np.unique(np.sort(np.stack([slice_of[a[cross]], slice_of[b[cross]]], 1), axis=1), axis=0)
    lengths = np.linalg.norm(centre[pairs[:, 0]] - centre[pairs[:, 1]], axis=1)
    nbrs, alive = _spanning(m, pairs[:, 0], pairs[:, 1], lengths)
    # how wide a slice is: a bump on the body grows from a wide slice, a leg from a narrow one
    width = np.maximum(radius, np.sqrt(counts / (np.pi * LAYER)))
    merged = np.arange(m)
    _prune(centre, width, nbrs, merged, SPUR_LAYER)
    keep = alive & (merged == np.arange(m))
    root_node = int(np.cumsum(keep)[slice_of[root]] - 1)
    return (*_compact(centre, radius, nbrs, keep), root_node)


def _graft(pts, rad, nbrs, lpts, lrad, lnbrs, lroot):
    """The thinned tree plus the branches of the layered one that it lost (a leg thinning wore away)."""
    from scipy.spatial import cKDTree

    if len(lpts) == 0 or len(pts) == 0:
        return pts, rad, nbrs
    dist, near = cKDTree(pts).query(lpts)
    covered = dist <= np.maximum(2.0, 1.5 * lrad)
    parent = np.full(len(lpts), -1)
    order, seen = [lroot], {lroot}
    for i in order:
        for j in lnbrs[i]:
            if j not in seen:
                seen.add(j)
                parent[j] = i
                order.append(j)
    base = len(pts)
    pts, rad, nbrs = list(pts), list(rad), [set(s) for s in nbrs]
    added: dict[int, int] = {}
    for e in (i for i in range(len(lpts)) if len(lnbrs[i]) == 1 and not covered[i]):
        path, x = [], e
        while x >= 0 and not covered[x] and x not in added:  # back towards the body until the thinned tree is there
            path.append(x)
            x = int(parent[x])
        if len(path) < 2 or np.linalg.norm(lpts[path[0]] - lpts[path[-1]]) < GRAFT_MIN:
            continue
        for k, node in enumerate(path):
            added[node] = len(pts)
            pts.append(lpts[node])
            rad.append(lrad[node])
            nbrs.append(set())
            if k:
                nbrs[added[node]].add(added[path[k - 1]])
                nbrs[added[path[k - 1]]].add(added[node])
        if x in added:
            joint = added[x]
        elif x >= 0:
            joint = int(near[x])
        else:
            joint = int(cKDTree(np.asarray(pts[:base])).query(lpts[path[-1]])[1])
        nbrs[added[path[-1]]].add(joint)
        nbrs[joint].add(added[path[-1]])
    return np.asarray(pts, np.float64), np.asarray(rad, np.float64), nbrs


def _prune(centre: np.ndarray, width: np.ndarray, nbrs: list[set[int]], merged: np.ndarray, spur: float) -> None:
    """Cut twigs shorter than ``spur`` half-widths of the node they grow from, until none is left; a cut node
    belongs to that node (``merged``)."""
    while True:
        cut = False
        for i in range(len(nbrs)):
            if len(nbrs[i]) != 1 or merged[i] != i:
                continue
            twig, prev, cur = [i], i, next(iter(nbrs[i]))
            length = np.linalg.norm(centre[i] - centre[cur])
            while len(nbrs[cur]) == 2:
                twig.append(cur)
                prev, cur = cur, next(x for x in nbrs[cur] if x != prev)
                length += np.linalg.norm(centre[twig[-1]] - centre[cur])
            if len(nbrs[cur]) >= 3 and length < max(SPUR_MIN, spur * width[cur]):
                for t in twig:
                    for q in nbrs[t]:
                        nbrs[q].discard(t)
                    nbrs[t] = set()
                    merged[t] = cur
                cut = True
        if not cut:
            return
