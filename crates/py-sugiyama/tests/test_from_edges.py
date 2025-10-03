import networkx as nx
import pytest

from fast_sugiyama import from_edges
from fast_sugiyama._types import PositionsType


def check_positions(positions: PositionsType, fast=True) -> bool:
    reg = {(x, y) for _n, (x, y) in positions}
    is_valid = len(reg) == len(positions)

    if fast:
        return is_valid

    else:  # pragma: no cover
        reg = set()
        for _n, val in positions:
            if val in reg:
                x, y = val
                msg = (
                    "Duplicate coordinate found!\n",
                    f"\t node id: {_n} @ ({x}, {y})\n",
                    f"\tPositions: {positions}",
                )
                raise ValueError(msg)
            reg.add(val)
        return True


def test_check_positions():
    pos = [(0, (5.0, 0.0)), (1, (3.0, 0.0)), (2, (5.0, 0.0))]
    assert not check_positions(pos)


@pytest.mark.parametrize(
    "edges",
    [
        [
            (1, 0),
            (2, 1),
            (3, 1),
            (4, 3),
            (5, 1),
            (6, 2),
            (7, 2),
            (8, 4),
            (9, 8),
            (10, 7),
            (11, 4),
            (12, 7),
            (13, 6),
            (14, 4),
            (15, 4),
            (16, 8),
            (17, 16),
            (18, 1),
            (19, 2),
            (20, 7),
            (21, 16),
            (22, 16),
            (23, 1),
            (24, 16),
            (25, 4),
            (26, 2),
            (27, 25),
            (28, 1),
            (29, 28),
            (30, 11),
            (31, 24),
            (32, 13),
            (33, 4),
            (34, 4),
        ],
        [
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (3, 8),
            (3, 9),
            (4, 10),
            (5, 11),
            (5, 12),
            (6, 13),
        ],
        [
            (24, 12),
            (12, 2),
            (14, 2),
            (202, 8),
            (8, 6),
            (203, 8),
            (204, 8),
            (6, 2),
            (101, 2),
            (104, 2),
            (16, 12),
            (18, 12),
            (102, 2),
            (103, 2),
            (20, 8),
            (10, 2),
            (22, 6),
        ],
    ],
)
@pytest.mark.parametrize("reversed", [False, True])
def test_from_edges(edges, reversed):
    if reversed:
        edges = [(t, s) for s, t in edges]

    layouts = from_edges(edges)

    for positions, *_ in layouts:
        assert check_positions(positions)


@pytest.mark.parametrize("_n", range(5))
def test_large_graph(_n):
    edges = list(nx.gn_graph(1000, create_using=nx.DiGraph).edges())
    layouts = from_edges(edges, vertex_spacing=10)
    for positions, *_ in layouts:
        if not check_positions(positions):  # pragma: no cover
            assert check_positions(positions, fast=False)


@pytest.mark.parametrize("n", range(10, 161, 30))
def test_tiny(n):
    ng = 25

    for ctype in ["median", "barycenter"]:
        for i in range(ng):
            edges = list(
                nx.gn_graph(n, seed=n + i * 42, create_using=nx.DiGraph).edges()
            )

            layouts = from_edges(edges, crossing_minimization=ctype)  # type: ignore

            for positions, *_ in layouts:
                if not check_positions(positions):  # pragma: no cover
                    print(f"ctype: {ctype}, iters: {i}, len: {n}")
                    print(f"edges: {edges}")
                    print(f"positions {positions}")
                    assert check_positions(positions, fast=False)
