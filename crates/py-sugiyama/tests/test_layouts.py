import pytest

from fast_sugiyama import from_edges, layout


@pytest.mark.parametrize("spacing", [None, 10])
def test_layout_transitivity(multi_graph, spacing):
    orig_nodes = list(multi_graph.nodes())

    layouts = from_edges(list(multi_graph.edges()))
    pos = layouts.to_dict()
    assert all(n in pos for n in orig_nodes)

    layouts = (
        layouts.align_layouts_horizontal(spacing)
        .align_layouts_horizontal_center()
        .align_layouts_vertical(spacing)
        .shuffle()
        .align_layouts_vertical_center()
        .align_layouts_vertical_top()
        .shuffle(seed=10)
        .compact_layouts_horizontal(spacing)
        .rect_pack_layouts(spacing)
        .sort_vertical()
        .compact_layout(spacing)
        .sort_horizontal()
        .dot_layout(spacing)
        ._check_wh()
        .rect_pack_layouts(max_width=5000)
        .rect_pack_layouts(max_height=5000)
    )

    pos = layouts.to_dict()
    assert all(n in pos for n in orig_nodes)

    assert (
        len(layouts.to_bboxes())
        == len(layouts)
        == len(layouts.to_list())
        == len(layouts.get_origins())
    )
    bboxes = layouts.to_bboxes(20)
    assert len(bboxes) == len(layouts)

    assert 5 == len(layouts[:5])
    assert len(layouts[0]) > 0
    assert len(layouts.flatten_positions()) > 0


@pytest.mark.parametrize(
    "x, base, exp",
    [
        (5.1, None, 5),
        (30.1, 50, 50),
        (5, 5, 5),
        (0, -10, 0),
        (6, -10, 10),
        (-7.5, 2, -8),
    ],
)
def test_round(x, base, exp):
    assert exp == layout._round(x, base)
