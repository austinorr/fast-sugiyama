# `Fast-Sugiyama`

This project adds python bindings for the [`rust-sugiyama`](https://crates.io/crates/rust-sugiyama) crate, to produce directed graph layouts similar to the GraphViz `dot` program.

<div align="center" >
    <img width=400 src="https://raw.githubusercontent.com/austinorr/fast-sugiyama/main/crates/py-sugiyama/misc/hero.png" alt="Graph Example">
</div>

## Description

An implementation of Sugiyama's algorithm for displaying a layered graph.

## Quick Start

```python
import networkx as nx
from fast_sugiyama import from_edges

g = nx.gn_graph(42, seed=132, create_using=nx.DiGraph)
pos = from_edges(g.edges()).to_dict()
nx.draw_networkx(g, pos=pos, with_labels=False, node_size=150)

```

<div align="left" >
    <img width=500 src="https://raw.githubusercontent.com/austinorr/fast-sugiyama/main/crates/py-sugiyama/misc/quickstart.png" alt="Quick Start Output">
</div>
