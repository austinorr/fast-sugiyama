from typing import Literal

from ._fast_sugiyama import from_edges as _from_edges
from ._types import EdgeListType, NumType
from .layout import PYDOT_SPACING, Layouts

Ranking = Literal[
    "original",
    "minimize",
    "up",
    "down",
]

CrossingMinimization = Literal["median", "barycenter"]


def from_edges(
    edges: EdgeListType,
    minimum_length: NumType | None = None,
    vertex_spacing: NumType | None = None,
    dummy_vertices: bool | None = None,
    dummy_size: NumType | None = None,
    ranking_type: Ranking | None = None,
    crossing_minimization: CrossingMinimization | None = None,
    transpose: bool | None = None,
    check_layout: bool | None = None,
) -> Layouts:
    if vertex_spacing is None:
        vertex_spacing = PYDOT_SPACING

    # The rust impl only accepts edges with source & target nodes stored as usize/int.
    # First, get unique nodes as list. We'll use the list index to translate from
    # usize to the original at the end.
    nodes = list({n for edge in edges for n in edge})

    # Map node to an int. Here we are just using its index in the set we created above.
    node_ints = dict(zip(nodes, range(len(nodes)), strict=True))

    # Build a new edge list which uses ints as the node id.
    edges_ints = [(node_ints[s], node_ints[t]) for s, t in edges]

    layouts_int = _from_edges(
        edges_ints,
        minimum_length=minimum_length,
        vertex_spacing=vertex_spacing,
        dummy_vertices=dummy_vertices,
        dummy_size=dummy_size,
        ranking_type=ranking_type,
        crossing_minimization=crossing_minimization,
        transpose=transpose,
        check_layout=check_layout,
        encode_edges=False,
    )

    layouts = Layouts()

    for positions_int, w, h, el in layouts_int:
        positions = []
        for n_int, xy in positions_int:
            nid = n_int

            # only branches if dummies vertices are included in layout
            if n_int < len(nodes):  # pragma: no branch
                nid = nodes[n_int]
            positions.append((nid, xy))

        if el is not None:
            el_remap = []
            for s, t in el:
                if s < len(nodes):
                    s = nodes[s]
                if t < len(nodes):
                    t = nodes[t]
                el_remap.append((s, t))
            el = el_remap

        layouts.append((positions, w, h, el))

    return layouts
