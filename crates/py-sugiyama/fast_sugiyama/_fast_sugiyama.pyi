from ._types import (
    CrossingMinimization,
    LayoutsIntType,
    NumType,
    Ranking,
)

def _from_edges(
    edges: list[tuple[int, int]],
    minimum_length: NumType | None = None,
    vertex_spacing: NumType | None = None,
    dummy_vertices: bool | None = None,
    dummy_size: NumType | None = None,
    ranking_type: Ranking | None = None,
    crossing_minimization: CrossingMinimization | None = None,
    transpose: bool | None = None,
    check_layout: bool | None = None,
    encode_edges: bool | None = None,
) -> LayoutsIntType: ...
