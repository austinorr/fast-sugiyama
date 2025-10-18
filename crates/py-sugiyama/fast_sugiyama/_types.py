from typing import Iterable, Literal, Sequence

NumType = int | float
NodeIDType = int | str
EdgeType = tuple[NodeIDType, NodeIDType]  # we support mixed type edges
EdgeListType = Iterable[EdgeType]
EdgeListIntType = Iterable[tuple[int, int]]
CoordType = tuple[float, float] | tuple[int, int]
PositionType = tuple[NodeIDType, CoordType]
PositionIntType = tuple[int, CoordType]
PositionsType = Sequence[PositionType]
PositionsIntType = Sequence[PositionIntType]
LayersType = dict[NumType, Sequence[NodeIDType]]
BBoxesType = Sequence[tuple[CoordType, NumType, NumType]]
LayoutType = tuple[PositionsType, NumType, NumType, EdgeListType | None]
LayoutIntType = tuple[PositionsIntType, NumType, NumType, EdgeListIntType | None]
LayoutsType = list[LayoutType]
LayoutsIntType = list[LayoutIntType]

Ranking = Literal[
    "original",
    "minimize",
    "up",
    "down",
]

CrossingMinimization = Literal["median", "barycenter"]
