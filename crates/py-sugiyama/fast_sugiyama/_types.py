from typing import Sequence

NumType = int | float
NodeIDType = int | str
EdgeType = tuple[NodeIDType, NodeIDType]  # we support mixed type edges
EdgeListType = Sequence[EdgeType]
CoordType = tuple[float, float] | tuple[int, int]
PositionType = tuple[NodeIDType, CoordType]
PositionsType = Sequence[PositionType]
LayersType = dict[NumType, Sequence[NodeIDType]]
BBoxesType = Sequence[tuple[CoordType, NumType, NumType]]
LayoutType = tuple[PositionsType, NumType, NumType, EdgeListType | None]
LayoutsType = list[LayoutType]
