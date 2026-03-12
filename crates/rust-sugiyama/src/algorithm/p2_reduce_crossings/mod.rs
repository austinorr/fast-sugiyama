#[cfg(test)]
mod tests;
use std::collections::HashSet;
use std::fmt::Display;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};

use log::{debug, info, trace};
use petgraph::Direction::{Incoming, Outgoing};
use petgraph::algo::toposort;
use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use petgraph::visit::NodeIndexable;

use crate::configure::{CROSSING_LOG_TARGET, CrossingMinimization};
use crate::util::{IterDir, iterate, radix_sort};

use super::{Edge, Vertex, slack};

#[derive(Clone)]
struct Order {
    _inner: Vec<Vec<NodeIndex>>,
    /// Indexed by `NodeIndex::index()`. Sized to `graph.node_bound()` so every
    /// valid NodeIndex maps directly without hashing.
    positions: Vec<usize>,
    node_bound: usize,
}

impl Display for Order {
    #[cfg_attr(coverage, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for row in &self._inner {
            for c in row {
                s.push_str(&c.index().to_string());
                s.push(',')
            }
            s.push('\n');
        }
        f.write_str(&s)
    }
}

impl Order {
    fn new(layers: Vec<Vec<NodeIndex>>, node_bound: usize) -> Self {
        let mut positions = vec![0usize; node_bound];
        for l in &layers {
            for (pos, v) in l.iter().enumerate() {
                positions[v.index()] = pos;
            }
        }
        Self {
            _inner: layers,
            positions,
            node_bound,
        }
    }

    fn max_rank(&self) -> usize {
        self.len()
    }

    fn exchange(&mut self, a: usize, b: usize, r: usize) {
        // first update positions, then swap
        self.positions[self._inner[r][a].index()] = b;
        self.positions[self._inner[r][b].index()] = a;
        self._inner[r].swap(a, b);
    }

    fn cross_count_two_vertices(
        &self,
        v: NodeIndex,
        w: NodeIndex,
        graph: &StableDiGraph<Vertex, Edge>,
        v_adjacent: &mut Vec<usize>,
        w_adjacent: &mut Vec<usize>,
    ) -> usize {
        let mut crossings = 0;
        for dir in [Incoming, Outgoing] {
            v_adjacent.clear();
            v_adjacent.extend(
                graph
                    .neighbors_directed(v, dir)
                    .map(|n| self.positions[n.index()]),
            );
            v_adjacent.sort_unstable();

            w_adjacent.clear();
            w_adjacent.extend(
                graph
                    .neighbors_directed(w, dir)
                    .map(|n| self.positions[n.index()]),
            );
            w_adjacent.sort_unstable();

            crossings += Self::calculate_cross_count_two_vertices(v_adjacent, w_adjacent);
        }
        crossings
    }

    fn calculate_cross_count_two_vertices(v_adjacent: &[usize], w_adjacent: &[usize]) -> usize {
        let mut all_crossings = 0;
        let mut k = 0;
        for i in v_adjacent {
            let i = *i;
            let mut crossings = k;
            while k < w_adjacent.len() && w_adjacent[k] < i {
                let j = w_adjacent[k];
                if i > j {
                    crossings += 1;
                }
                k += 1;
            }
            all_crossings += crossings;
        }
        all_crossings
    }

    fn crossings(&self, graph: &StableDiGraph<Vertex, Edge>) -> usize {
        let mut cross_count = 0;
        for rank in 0..self.max_rank() - 1 {
            cross_count += self.bilayer_cross_count(graph, rank);
        }
        cross_count
    }

    fn bilayer_cross_count(&self, graph: &StableDiGraph<Vertex, Edge>, rank: usize) -> usize {
        // find initial edge order
        let north = &self[rank];
        let south = &self[rank + 1];
        let mut len = south.len();
        let mut key_length = 0;
        while len > 0 {
            len /= 10;
            key_length += 1;
        }
        let edge_endpoint_positions = north
            .iter()
            .flat_map(|v| {
                radix_sort(
                    graph
                        .neighbors_directed(*v, Outgoing)
                        .filter(|n| graph[*v].rank.abs_diff(graph[*n].rank) == 1)
                        .map(|n| self.positions[n.index()])
                        .collect(),
                    key_length,
                )
            })
            .collect::<Vec<_>>();
        Self::count_crossings(edge_endpoint_positions, south.len())
    }

    fn count_crossings(endpoints: Vec<usize>, south_len: usize) -> usize {
        // build the accumulator tree
        let mut c = 0;
        while 1 << c < south_len {
            c += 1
        }
        let tree_size = (1 << (c + 1)) - 1;
        let first_index = (1 << c) - 1;
        let mut tree = vec![0; tree_size];

        let mut cross_count = 0;

        // traverse through the positions and adjust tree nodes
        for pos in endpoints {
            let mut index = pos + first_index;
            tree[index] += 1;
            while index > 0 {
                // traverse up the tree, incrementing the nodes of the tree
                // each time we visit them.
                //
                // When visiting a left node, add the value of the node on the right to
                // the cross count;
                if index % 2 == 1 {
                    cross_count += tree[index + 1]
                }
                index = (index - 1) / 2;
                tree[index] += 1;
            }
        }
        cross_count
    }

    #[cfg_attr(coverage, coverage(off))]
    #[allow(dead_code)]
    fn print(&self) {
        for line in &self._inner {
            for v in line {
                print!("{v:>2?} ");
            }
            println!();
        }
    }
}

impl Deref for Order {
    type Target = Vec<Vec<NodeIndex>>;

    fn deref(&self) -> &Self::Target {
        &self._inner
    }
}

impl DerefMut for Order {
    #[cfg_attr(coverage, coverage(off))]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self._inner
    }
}

pub(super) fn insert_dummy_vertices(
    graph: &mut StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
    dummy_size: f64,
    dummy_id_counter: &AtomicUsize,
) {
    // find all edges that have slack of greater than 0.
    // and insert dummy vertices
    info!(target: CROSSING_LOG_TARGET, "Inserting dummy vertices for edges spanning more than {minimum_length} ranks");
    for edge in graph.edge_indices().collect::<Vec<_>>() {
        if slack(graph, edge, minimum_length) > 0 {
            let (mut tail, head) = graph.edge_endpoints(edge).unwrap();
            trace!(target: CROSSING_LOG_TARGET,
                "Inserting {} dummy vertices between: ({}, {})",
                graph[head].rank - graph[tail].rank - 1,
                tail.index(),
                head.index());

            if let Some(w) = graph.remove_edge(edge) {
                for rank in (graph[tail].rank + 1)..graph[head].rank {
                    // usize usize::MAX id as reserved value for a dummy vertex
                    let d = Vertex {
                        is_dummy: true,
                        size: (dummy_size, 0.0),
                        ..Default::default()
                    };
                    let new = graph.add_node(d);
                    graph[new].id = dummy_id_counter.fetch_add(1, Ordering::Relaxed);
                    graph[new].align = new;
                    graph[new].root = new;
                    graph[new].sink = new;
                    graph[new].rank = rank;
                    graph.add_edge(tail, new, w); // naively propagate weight to dummy edges
                    tail = new;
                }
                graph.add_edge(tail, head, w); // add last dummy edge connecting to the head
            }
        }
    }
}

pub(super) fn remove_dummy_vertices(
    graph: &mut StableDiGraph<Vertex, Edge>,
    order: &mut [Vec<NodeIndex>],
) {
    // go through all nodes in topological order
    // see if any outgoing neighbors are dummies
    // follow them until the other non dummy node is found
    // insert old edge
    // remove all dummy nodes
    info!(target: CROSSING_LOG_TARGET, "Removing dummy vertices and inserting original edges.");
    let vertices = toposort(&*graph, None).unwrap();
    for v in vertices {
        let mut edges = Vec::new();
        for mut n in graph.neighbors_directed(v, Outgoing) {
            if graph[n].is_dummy {
                while graph[n].is_dummy {
                    let dummy_neighbors = graph.neighbors_directed(n, Outgoing).collect::<Vec<_>>();
                    //assert_eq!(dummy_neighbors.len(), 1);
                    n = dummy_neighbors[0];
                }
                edges.push((v, n));
            }
        }
        for (tail, head) in edges {
            graph.add_edge(tail, head, Edge::default());
        }
    }
    // remove from order
    for l in order {
        l.retain(|v| !graph[*v].is_dummy);
    }
    graph.retain_nodes(|g, v| !g[v].is_dummy);
}

// TODO: Maybe write store all upper neighbors on vertex directly
pub(super) fn ordering(
    graph: &mut StableDiGraph<Vertex, Edge>,
    crossing_minimization: CrossingMinimization,
    transpose: bool,
) -> Vec<Vec<NodeIndex>> {
    let order = init_order(graph);
    // move downwards for crossing reduction
    let cm_method = match crossing_minimization {
        CrossingMinimization::Barycenter => self::barycenter,
        CrossingMinimization::Median => self::median,
    };
    let order = reduce_crossings_bilayer_sweep(graph, order, cm_method, transpose);
    order._inner
}

type CMMethod = fn(&StableDiGraph<Vertex, Edge>, NodeIndex, bool, &[usize]) -> f64;

fn init_order(graph: &StableDiGraph<Vertex, Edge>) -> Order {
    info!(target: CROSSING_LOG_TARGET,
        "Initializing order of vertices in each rank via dfs.");

    fn dfs(
        v: NodeIndex,
        order: &mut Vec<Vec<NodeIndex>>,
        graph: &StableDiGraph<Vertex, Edge>,
        visited: &mut HashSet<NodeIndex>,
    ) {
        if !visited.contains(&v) {
            visited.insert(v);
            order[graph[v].rank as usize].push(v);
            graph
                .neighbors_directed(v, Outgoing)
                .for_each(|n| dfs(n, order, graph, visited))
        }
    }

    let max_rank = graph
        .node_weights()
        .map(|v| v.rank as usize)
        .max_by(|r1, r2| r1.cmp(r2))
        .expect("Got invalid ranking");
    let mut order = vec![Vec::new(); max_rank + 1];
    let mut visited = HashSet::new();

    // build initial order via dfs
    graph
        .node_indices()
        .for_each(|v| dfs(v, &mut order, graph, &mut visited));

    Order::new(order, graph.node_bound())
}

fn reduce_crossings_bilayer_sweep(
    graph: &StableDiGraph<Vertex, Edge>,
    mut order: Order,
    cm_method: CMMethod,
    transpose: bool,
) -> Order {
    info!(target: CROSSING_LOG_TARGET, "Reducing crossings via bilayer sweep");
    let node_bound = order.node_bound;
    let mut best_crossings = order.crossings(graph);
    debug!(target: CROSSING_LOG_TARGET, "Initial number of crossings: {best_crossings}");
    let mut last_best = 0;
    let mut best_layers = order._inner.clone();
    for i in 0.. {
        let move_down = i % 2 == 0;
        order = order_layer(graph, &order, move_down, cm_method);
        if transpose {
            self::transpose(graph, &mut order, move_down);
        }
        let crossings = order.crossings(graph);
        trace!(target: CROSSING_LOG_TARGET, "Current number of crossings: {crossings}");
        if crossings < best_crossings {
            best_crossings = crossings;
            debug!(target: CROSSING_LOG_TARGET, "Lowest number of crossings so far: {best_crossings}");
            best_layers.clone_from(&order._inner);
            last_best = 0;
        } else {
            last_best += 1;
        }
        if last_best == 4 {
            info!(target: CROSSING_LOG_TARGET, "Didn't improve after 4 sweeps, returning");
            return Order::new(best_layers, node_bound);
        }
    }
    Order::new(best_layers, node_bound)
}

fn transpose(graph: &StableDiGraph<Vertex, Edge>, order: &mut Order, move_down: bool) {
    trace!(target: CROSSING_LOG_TARGET,
        "Using transpose, try to swap vertices in each layer manually to reduce cross count");

    let mut improved = true;
    let iter_dir = if move_down {
        IterDir::Forward
    } else {
        IterDir::Backward
    };

    // Allocate scratch buffers once for the entire transpose phase
    let mut v_adjacent: Vec<usize> = Vec::with_capacity(64);
    let mut w_adjacent: Vec<usize> = Vec::with_capacity(64);

    while improved {
        improved = false;
        for r in iterate(iter_dir, order.max_rank()) {
            trace!(target: CROSSING_LOG_TARGET, "Transpose vertices in rank {r}");
            for i in 0..order._inner[r].len() - 1 {
                let v = order._inner[r][i];
                let w = order._inner[r][i + 1];
                let v_w_crossing =
                    order.cross_count_two_vertices(v, w, graph, &mut v_adjacent, &mut w_adjacent);
                let w_v_crossing =
                    order.cross_count_two_vertices(w, v, graph, &mut v_adjacent, &mut w_adjacent);
                if v_w_crossing > w_v_crossing {
                    improved = true;
                    order.exchange(i, i + 1, r);
                }
            }
        }
        trace!(target: CROSSING_LOG_TARGET, "Did improve: {improved}");
    }
}

fn order_layer(
    graph: &StableDiGraph<Vertex, Edge>,
    cur_order: &Order,
    move_down: bool,
    cm_method: CMMethod,
) -> Order {
    let node_bound = cur_order.node_bound;
    let mut new_order = vec![Vec::new(); cur_order.max_rank()];
    let mut positions = cur_order.positions.clone();
    let dir: Vec<usize> = if move_down {
        new_order[0].clone_from(&cur_order._inner[0]);
        (1..cur_order.max_rank()).collect()
    } else {
        new_order[cur_order.max_rank() - 1].clone_from(&cur_order._inner[cur_order.max_rank() - 1]);
        (0..cur_order.max_rank() - 1).rev().collect()
    };

    // Reusable score buffer, cleared per-rank implicitly by overwriting
    let mut scores = vec![0.0f64; node_bound];

    for rank in dir {
        trace!(target: CROSSING_LOG_TARGET, "Updating order of vertices in rank {rank}");
        trace!(target: CROSSING_LOG_TARGET, "Original order: {:?}",
            cur_order[rank].iter().map(|v| v.index()).collect::<Vec<_>>()
        );

        new_order[rank].clone_from(&cur_order[rank]);
        // Fill scores for each node in this rank, then sort by score
        for &n in &new_order[rank] {
            scores[n.index()] = cm_method(graph, n, move_down, &positions);
        }
        new_order[rank].sort_unstable_by(|a, b| scores[a.index()].total_cmp(&scores[b.index()]));

        new_order[rank].iter().enumerate().for_each(|(pos, v)| {
            positions[v.index()] = pos;
        });
        trace!(target: CROSSING_LOG_TARGET, "Updated order : {:?}",
            new_order[rank].iter().map(|v| v.index()).collect::<Vec<_>>()
        );
    }

    Order::new(new_order, node_bound)
}

fn barycenter(
    graph: &StableDiGraph<Vertex, Edge>,
    vertex: NodeIndex,
    move_down: bool,
    positions: &[usize],
) -> f64 {
    let neighbors: Vec<_> = if move_down {
        graph.neighbors_directed(vertex, Incoming).collect()
    } else {
        graph.neighbors_directed(vertex, Outgoing).collect()
    };

    if neighbors.is_empty() {
        return positions[vertex.index()] as f64;
    }

    // Only look at direct neighbors
    let adjacent = neighbors
        .into_iter()
        // .filter(|n| graph[vertex].rank.abs_diff(graph[*n].rank) == 1)
        .map(|n| positions[n.index()])
        .collect::<Vec<usize>>();

    if !adjacent.is_empty() {
        adjacent.iter().sum::<usize>() as f64 / adjacent.len() as f64
    } else {
        0.0
    }
}

fn median(
    graph: &StableDiGraph<Vertex, Edge>,
    vertex: NodeIndex,
    move_down: bool,
    positions: &[usize],
) -> f64 {
    let neighbors: Vec<_> = if move_down {
        graph.neighbors_directed(vertex, Incoming).collect()
    } else {
        graph.neighbors_directed(vertex, Outgoing).collect()
    };
    // Only look at direct neighbors
    let mut adjacent = neighbors
        .into_iter()
        .filter(|n| graph[vertex].rank.abs_diff(graph[*n].rank) == 1)
        .map(|n| positions[n.index()])
        .collect::<Vec<_>>();

    adjacent.sort();

    let length_p = adjacent.len();
    let m = length_p / 2;
    if length_p == 0 {
        f64::MAX
    } else if length_p % 2 == 1 {
        adjacent[m] as f64
    } else if length_p == 2 {
        (adjacent[0] + adjacent[1]) as f64 / 2.
    } else {
        let left = adjacent[m - 1] - adjacent[0];
        let right = adjacent[length_p - 1] - adjacent[m];
        let denom = left + right;
        if denom != 0 {
            (adjacent[m - 1] * right + adjacent[m] * left) as f64 / denom as f64
        } else {
            trace!(target: CROSSING_LOG_TARGET, "Computing Median divided by zero. adjacent {adjacent:?}");
            0.0
        }
    }
}
