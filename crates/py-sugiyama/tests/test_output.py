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


@pytest.mark.output
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

    fig.savefig(misc / "pydot_compare.jpg", dpi=150)


@pytest.mark.output
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

    fig.savefig(misc / "difficult_pydot.jpg", dpi=150)


@pytest.mark.output
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

    fig.savefig(misc / "rect_pack_layout.jpg", dpi=150)


@pytest.mark.output
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

    fig.savefig(misc / "compact_layout.jpg", dpi=150)
