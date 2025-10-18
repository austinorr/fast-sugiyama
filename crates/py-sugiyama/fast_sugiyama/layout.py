import random
from collections import defaultdict
from typing import Self, SupportsIndex, overload

from ._types import (
    BBoxesType,
    CoordType,
    LayersType,
    LayoutsType,
    LayoutType,
    NodeIDType,
    NumType,
    PositionsType,
    Sequence,
)

PYDOT_SPACING = 72  # approximately


def _round(x: NumType, base: int | None = None):
    if not base:  # handle base = 0 & base = None
        base = 1
    base = abs(base)
    return base * round(x / base)


def flatten_positions(layouts: LayoutsType) -> PositionsType:
    return [(n, xy) for pos, *_ in layouts for n, xy in pos]


def to_dict(layouts: LayoutsType) -> dict[NodeIDType, CoordType]:
    return dict(flatten_positions(layouts))


def build_layers(positions: PositionsType) -> LayersType:
    layers = defaultdict(list)
    for n, (_, y) in positions:
        layers[y].append(n)
    return dict(layers)


def get_position_origin(positions: PositionsType) -> CoordType:
    x = min(x for _, (x, _) in positions)
    y = min(y for _, (_, y) in positions)
    return x, y


def get_origin(layout: LayoutType) -> CoordType:
    pos, *_ = layout
    return get_position_origin(pos)


def get_origins(layouts: LayoutsType) -> Sequence[CoordType]:
    return [get_position_origin(pos) for pos, *_ in layouts]


def get_bboxes(layouts: LayoutsType, spacing: NumType | None = None) -> BBoxesType:
    """Create boxes for use in mpl.Rectangle. This is useful for debugging."""
    if spacing is None:
        spacing = PYDOT_SPACING
    origins = get_origins(layouts)

    bboxes = [
        ((x - spacing / 2, y - spacing / 2), width + spacing, height + spacing)
        for (x, y), (width, height) in zip(
            origins, [(w, h) for _, w, h, *_ in layouts], strict=True
        )
    ]

    return bboxes


class Layouts(LayoutsType):
    def __init__(self, data: LayoutsType | None = None):
        super().__init__()
        if data:
            self.extend(data)

    @overload
    def __getitem__(self, key: SupportsIndex) -> LayoutType: ...

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    def __getitem__(self, key: slice | SupportsIndex) -> LayoutType | LayoutsType:
        if isinstance(key, slice):
            datas: LayoutsType = Layouts(list.__getitem__(self, key))
            return datas
        data: LayoutType = list.__getitem__(self, key)
        return data

    def to_dict(self):
        return to_dict(self)

    def to_list(self) -> LayoutsType:
        return self

    def flatten_positions(self) -> PositionsType:
        return flatten_positions(self)

    def get_origins(self):
        return get_origins(self)

    def to_bboxes(self, spacing: NumType | None = None):
        return get_bboxes(self, spacing=spacing)

    def shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self)
        return self

    def sort_horizontal(self, reverse: bool = False):
        return Layouts(sorted(self, key=lambda x: get_origin(x)[0], reverse=reverse))

    def sort_vertical(self, reverse: bool = False):
        return Layouts(sorted(self, key=lambda x: get_origin(x)[1], reverse=reverse))

    def align_layouts_vertical(self, spacing: NumType | None = None):
        return align_layouts_vertical(self, spacing=spacing)

    def align_layouts_horizontal(self, spacing: NumType | None = None):
        return align_layouts_horizontal(self, spacing=spacing)

    def align_layouts_vertical_top(self):
        return align_layouts_vertical_top(self)

    def align_layouts_vertical_center(self):
        return align_layouts_vertical_center(self)

    def align_layouts_horizontal_center(self):
        return align_layouts_horizontal_center(self)

    def compact_layouts_horizontal(self, spacing: NumType | None = None):
        return compact_layouts_horizontal(self, spacing=spacing)

    def rect_pack_layouts(
        self,
        spacing: NumType | None = None,
        max_width: NumType | None = None,
        max_height: NumType | None = None,
    ):
        return rect_pack_layouts(
            self, spacing=spacing, max_width=max_width, max_height=max_height
        )

    def dot_layout(self, spacing: NumType | None = None):
        return self.align_layouts_vertical_top().compact_layouts_horizontal(
            spacing=spacing
        )

    def compact_layout(
        self,
        spacing: NumType | None = None,
        max_width: NumType | None = None,
        max_height: NumType | None = None,
    ):
        return (
            self.rect_pack_layouts(
                spacing=spacing, max_width=max_width, max_height=max_height
            )
            .sort_horizontal()
            .compact_layouts_horizontal(spacing=spacing)
        )

    def _check_wh(self):
        for i, (pos, w, h, _) in enumerate(self):
            assert w > 0 and h > 0, (i, pos, w, h)
        return self


def align_layouts_horizontal(
    layouts: LayoutsType,
    spacing: NumType | None = None,
) -> Layouts:
    if spacing is None:
        spacing = PYDOT_SPACING
    new_layouts: LayoutsType = []
    offset = 0.0

    for positions, w, h, el in layouts:
        minx = min(x for _, (x, _) in positions)
        new_positions = [(n, (x - minx + offset, y)) for n, (x, y) in positions]
        new_layouts.append((new_positions, w, h, el))

        offset += w + spacing

    return Layouts(new_layouts)


def align_layouts_vertical(
    layouts: LayoutsType,
    spacing: NumType | None = None,
) -> Layouts:
    if spacing is None:
        spacing = PYDOT_SPACING
    new_layouts: LayoutsType = []
    offset = 0.0

    for positions, w, h, el in layouts:
        miny = min(y for _, (_, y) in positions)
        new_positions = [(n, (x, y - miny + offset)) for n, (x, y) in positions]
        new_layouts.append((new_positions, w, h, el))

        offset += h + spacing

    return Layouts(new_layouts)


def align_layouts_vertical_top(layouts: LayoutsType) -> Layouts:
    new_layouts: LayoutsType = []

    maxh = max(h for *_, h, _ in layouts)
    for positions, w, h, el in layouts:
        maxy = max(y for _, (_, y) in positions)
        diff = maxh - maxy
        new_positions = [(n, (x, y + diff)) for n, (x, y) in positions]
        new_layouts.append((new_positions, w, h, el))

    return Layouts(new_layouts)


def align_layouts_vertical_center(layouts: LayoutsType) -> Layouts:
    new_layouts: LayoutsType = []

    midh = max(h for *_, h, _ in layouts) / 2
    for positions, w, h, el in layouts:
        diff = midh - h / 2
        new_positions = [(n, (x, y + diff)) for n, (x, y) in positions]
        new_layouts.append((new_positions, w, h, el))

    return Layouts(new_layouts)


def align_layouts_horizontal_center(layouts: LayoutsType) -> Layouts:
    new_layouts: LayoutsType = []

    midw = max(w for _, w, *_ in layouts) / 2
    for positions, w, h, el in layouts:
        diff = midw - w / 2
        new_positions = [(n, (x + diff, y)) for n, (x, y) in positions]
        new_layouts.append((new_positions, w, h, el))

    return Layouts(new_layouts)


def compact_layouts_horizontal(
    layouts: LayoutsType,
    spacing: NumType | None = None,
) -> Layouts:
    if spacing is None:
        spacing = PYDOT_SPACING
    layers_xmax: dict[NumType, float] = defaultdict(float)
    new_layouts: LayoutsType = []

    for layout in layouts:
        pos_og, w, h, el = layout
        layers = build_layers(pos_og)

        pos = dict(pos_og)

        shifts = []
        for i, nodes in layers.items():
            l_min = min(pos[v][0] for v in nodes)
            shifts.append(layers_xmax[i] - l_min)

        shift = max(shifts)

        for n, (x, y) in pos.items():
            pos[n] = (x + shift, y)

        # store new xmax's for next round
        for i, nodes in layers.items():
            layers_xmax[i] = spacing + max(
                layers_xmax[i], max(pos[v][0] for v in nodes)
            )

        new_layouts.append((list(pos.items()), w, h, el))

    return Layouts(new_layouts)


def rect_pack_layouts(
    layouts: LayoutsType,
    spacing: NumType | None = None,
    max_width: NumType | None = None,
    max_height: NumType | None = None,
) -> Layouts:
    import rpack

    if spacing is None:
        spacing = PYDOT_SPACING

    max_value = max(max(w, h) for _, w, h, _ in layouts)
    resolution = 1000
    scale = resolution / max_value

    if max_width:
        max_width = int(max_width * scale)
    if max_height:
        max_height = int(max_height * scale)

    origins = rpack.pack(
        [
            (
                int(max((w + spacing) * scale, 1)),
                int(max((h + spacing) * scale, 1)),
            )
            for _, w, h, _ in layouts
        ],
        max_width=max_width,
        max_height=max_height,
    )

    # snap y's to nearest 'layer' spacing and then round both x's and y's to
    # nearest 0.5 increment to match the original layout
    origins = [
        (round(2 * x / scale) / 2, _round(2 * y / scale, int(round(spacing, 0))) / 2)
        for x, y in origins
    ]

    new_layouts: LayoutsType = []
    for (xo, yo), (positions, w, h, el) in zip(origins, layouts, strict=True):
        minx = min(x for _, (x, _) in positions)
        miny = min(y for _, (_, y) in positions)

        np = [(n, (x - minx + xo, y - miny + yo)) for n, (x, y) in positions]
        new_layouts.append((np, w, h, el))

    return Layouts(new_layouts)
