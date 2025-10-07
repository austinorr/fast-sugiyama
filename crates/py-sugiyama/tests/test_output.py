from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pytest

from fast_sugiyama import from_edges


@pytest.fixture(scope="session")
def misc() -> Path:
    test_dir = Path(__file__).parent
    misc = test_dir.parent / "misc"
    misc.mkdir(exist_ok=True, parents=True)
    return misc.resolve()


@pytest.mark.skipif(
    "not config.getoption('--save-output')",
    reason="Only run when --save-output is given",
)
def test_compare_pydot(misc, multi_graph):
    g = multi_graph

    nodes = [
        n
        for c in list(nx.weakly_connected_components(g))[:15]
        for n in c
        if len(c) < 25
    ]
    sg = g.subgraph(nodes)

    positions = [
        (
            'nx.nx_pydot.pydot_layout(sg, prog="dot"))',
            nx.nx_pydot.pydot_layout(sg, prog="dot"),
        ),
        (
            "from_edges(sg.edges()).dot_layout().to_dict()",
            from_edges(sg.edges()).dot_layout().to_dict(),
        ),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    for ax, (title, pos) in zip(axes, positions, strict=True):
        ax.set_title(title)
        ax.set_aspect("equal")
        nx.draw_networkx(sg, pos=pos, ax=ax, with_labels=False, node_size=30)

    fig.tight_layout()

    fig.savefig(misc / "pydot_compare.png", dpi=150)


@pytest.mark.skipif(
    "not config.getoption('--save-output')",
    reason="Only run when --save-output is given",
)
def test_difficult_pydot(misc, multi_graph):
    g = multi_graph

    positions = [
        (
            'nx.nx_pydot.pydot_layout(g, prog="dot"))',
            nx.nx_pydot.pydot_layout(g, prog="dot"),
        ),
        (
            "from_edges(g.edges()).dot_layout().to_dict()",
            from_edges(g.edges()).dot_layout().to_dict(),
        ),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 3), sharex=True, sharey=True)
    for ax, (title, pos) in zip(axes, positions, strict=True):
        ax.set_title(title)
        ax.set_aspect("equal")
        nx.draw_networkx(g, pos=pos, ax=ax, with_labels=False, node_size=30)

    fig.tight_layout()

    fig.savefig(misc / "difficult_pydot.png", dpi=150)


@pytest.mark.skipif(
    "not config.getoption('--save-output')",
    reason="Only run when --save-output is given",
)
def test_rect_pack_output(misc, multi_graph):
    g = multi_graph
    layout = from_edges(g.edges())
    rect_layout = layout.rect_pack_layouts(max_width=2500)
    pos = rect_layout.to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    for ax in axes:
        ax.set_aspect("equal")
        nx.draw_networkx(g, pos=pos, ax=ax, with_labels=False, node_size=30)

    ax = axes[0]
    for box in rect_layout.to_bboxes():
        ax.add_patch(plt.Rectangle(*box, fc="none", ec="lightgrey", zorder=-1))  # type: ignore

    fig.tight_layout()

    fig.savefig(misc / "rect_pack_layout.png", dpi=150)


@pytest.mark.skipif(
    "not config.getoption('--save-output')",
    reason="Only run when --save-output is given",
)
def test_compact_output(misc, multi_graph):
    g = multi_graph
    layout = from_edges(g.edges())
    compact_layout = layout.compact_layout(max_width=2500)
    pos = compact_layout.to_dict()

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_aspect("equal")

    nx.draw_networkx(g, pos=pos, ax=ax, with_labels=False, node_size=30)

    for box in compact_layout.to_bboxes():
        ax.add_patch(plt.Rectangle(*box, fc="none", ec="lightgrey", zorder=-1))  # type: ignore

    fig.tight_layout()

    fig.savefig(misc / "compact_layout.png", dpi=150)


@pytest.mark.skipif(
    "not config.getoption('--save-output')",
    reason="Only run when --save-output is given",
)
def test_quickstart(misc):
    g = nx.gn_graph(42, seed=132, create_using=nx.DiGraph)
    pos = from_edges(g.edges()).to_dict()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_aspect("equal")
    nx.draw_networkx(g, pos=pos, ax=ax, with_labels=False, node_size=150)
    fig.tight_layout()
    fig.savefig(misc / "quickstart.png", dpi=150)


@pytest.mark.skipif(
    "not config.getoption('--save-output')",
    reason="Only run when --save-output is given",
)
def test_hero(misc):
    edges = [
        (20, 17),
        (22, 20),
        (5, 1),
        (19, 0),
        (23, 22),
        (10, 0),
        (9, 8),
        (1, 0),
        (7, 1),
        (24, 17),
        (18, 16),
        (21, 6),
        (4, 2),
        (3, 0),
        (14, 0),
        (12, 6),
        (7, 22),
        (6, 1),
        (15, 1),
        (15, 7),
        (16, 6),
        (15, 10),
        (17, 7),
        (8, 1),
        (10, 4),
        (2, 0),
        (10, 1),
        (11, 9),
        (13, 0),
    ]

    layouts = from_edges(edges)
    pos = layouts.to_dict()
    nodes = list({n for edge in edges for n in edge})

    g = nx.DiGraph()
    for *_, el in layouts:
        if el is not None:
            g.add_edges_from(el)

    fig, ax = plt.subplots(figsize=(5, 4))

    nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=nodes)
    nx.draw_networkx_labels(g, pos, ax=ax, labels={n: n for n in nodes})

    _ = nx.draw_networkx_edges(
        g,
        pos,
        [e for e in g.edges() if e[1] not in nodes],
        arrows=False,
        width=1.25,
        ax=ax,
    )
    _ = nx.draw_networkx_edges(
        g,
        pos,
        [e for e in g.edges() if e[1] in nodes],
        node_size=[300 if n in nodes else 0 for n in g.nodes()],
        width=1.25,
        ax=ax,
    )

    ax.set_aspect("equal")
    fig.tight_layout()

    fig.savefig(misc / "hero.png", dpi=150)
