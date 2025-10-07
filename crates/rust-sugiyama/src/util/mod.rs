use crate::Layout;
use log::{debug, info};
use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use std::collections::{BTreeSet, HashMap, HashSet};
#[cfg(test)]
pub mod graph_generator;

pub fn weakly_connected_components<V: Copy, E: Copy>(
    graph: &StableDiGraph<V, E>,
) -> Vec<StableDiGraph<V, E>> {
    info!(target: "connected_components", "Splitting graph into its connected components");
    let mut components = Vec::new();
    let mut visited = HashSet::new();

    for node in graph.node_indices() {
        if visited.contains(&node) {
            continue;
        }

        let component_nodes = component_dfs(node, graph);
        let component = graph.filter_map(
            |n, w| {
                if component_nodes.contains(&n) {
                    Some(*w)
                } else {
                    None
                }
            },
            |_, w| Some(*w),
        );

        component_nodes.into_iter().for_each(|n| {
            visited.insert(n);
        });
        components.push(component);
    }
    debug!(target: "connected_components", "Found {} components", components.len());

    components
}

fn component_dfs<V: Copy, E: Copy>(
    start: NodeIndex,
    graph: &StableDiGraph<V, E>,
) -> HashSet<NodeIndex> {
    let mut queue = vec![start];
    let mut visited = HashSet::new();

    visited.insert(start);

    while let Some(cur) = queue.pop() {
        for neighbor in graph.neighbors_undirected(cur) {
            if visited.contains(&neighbor) {
                continue;
            }
            visited.insert(neighbor);
            queue.push(neighbor);
        }
    }

    visited
}

#[test]
fn into_weakly_connected_components_two_components() {
    let g = StableDiGraph::<usize, usize>::from_edges([(0, 1), (1, 2), (3, 2), (4, 5), (4, 6)]);
    let sgs = weakly_connected_components(&g);
    assert_eq!(sgs.len(), 2);
    assert!(sgs[0].contains_edge(0.into(), 1.into()));
    assert!(sgs[0].contains_edge(1.into(), 2.into()));
    assert!(sgs[0].contains_edge(3.into(), 2.into()));
    assert!(sgs[1].contains_edge(4.into(), 5.into()));
    assert!(sgs[1].contains_edge(4.into(), 6.into()));
}

// TODO: refactor into trait
// disable warnings, since we might still need this someday
#[allow(dead_code)]
pub(super) fn iterate(dir: IterDir, length: usize) -> impl Iterator<Item = usize> {
    let (mut start, step) = match dir {
        IterDir::Forward => (usize::MAX, 1), // up corresponds to left to right
        IterDir::Backward => (length, usize::MAX),
    };
    std::iter::repeat_with(move || {
        start = start.wrapping_add(step);
        start
    })
    .take(length)
}

#[allow(dead_code)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum IterDir {
    Forward,
    Backward,
}

// TODO: implement this with binary and see if it is faster
pub(super) fn radix_sort(mut input: Vec<usize>, key_length: usize) -> Vec<usize> {
    let mut output = vec![0; input.len()];

    let mut key = 1;
    for _ in 0..key_length {
        counting_sort(&mut input, key, &mut output);
        key *= 10;
    }
    input
}

#[inline(always)]
fn counting_sort(input: &mut [usize], key: usize, output: &mut [usize]) {
    let mut count = [0; 10];
    // insert initial counts
    for i in input.iter().map(|n| self::key(key, *n)) {
        count[i] += 1;
    }

    // built accumulative sum
    for i in 1..10 {
        count[i] += count[i - 1];
    }

    for i in (0..input.len()).rev() {
        let k = self::key(key, input[i]);
        count[k] -= 1;
        output[count[k]] = input[i];
    }
    input.copy_from_slice(&output[..input.len()]);
}

#[inline(always)]
fn key(key: usize, n: usize) -> usize {
    (n / key) % 10
}

#[test]
fn test_counting_sort_first_digit() {
    let mut input = [10, 0, 1, 5, 4, 22, 12];
    let mut output = [0; 7];
    counting_sort(&mut input, 1, &mut output);
    assert_eq!(output, [10, 0, 1, 22, 12, 4, 5])
}

#[test]
fn test_counting_sort_second_digit() {
    let mut input = [10, 0, 1, 5, 4, 22, 12];
    let mut output = [0; 7];
    counting_sort(&mut input, 10, &mut output);
    assert_eq!(output, [0, 1, 5, 4, 10, 12, 22]);
}

#[test]
fn test_radix_sort() {
    let input = [10, 0, 1, 5, 4, 22, 12];
    let output = radix_sort(input.to_vec(), 2);
    assert_eq!(output, [0, 1, 4, 5, 10, 12, 22]);
}

pub(super) fn encode_edges(edges: &[(u32, u32)]) -> (Vec<(u32, u32)>, Vec<u32>) {
    // This is the decoder. We get the decoded value by indexing into this vec.
    let nodes: Vec<u32> = edges
        .iter()
        // We preserve the order by using a BTree.
        .fold(BTreeSet::new(), |mut acc, p| {
            acc.insert(p.0);
            acc.insert(p.1);
            acc
        })
        .into_iter()
        .collect();

    let node_encoder: HashMap<u32, u32> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (*n, i as u32))
        .collect();

    let encoded_edges = edges
        .iter()
        .map(|(s, t)| (*node_encoder.get(s).unwrap(), *node_encoder.get(t).unwrap()))
        .collect::<Vec<_>>();
    (encoded_edges, nodes)
}

pub(super) fn decode_positions(pos: &mut [(usize, (f64, f64))], nodes: &[u32]) {
    for (n, _) in pos.iter_mut() {
        if *n < nodes.len() {
            *n = nodes[*n] as usize;
        }
    }
}

pub(super) fn decode_edges(edges: &mut [(usize, usize)], nodes: &[u32]) {
    for (s, t) in edges.iter_mut() {
        if *s < nodes.len() {
            *s = nodes[*s] as usize;
        }

        if *t < nodes.len() {
            *t = nodes[*t] as usize;
        }
    }
}

#[test]
fn test_encode_edges() {
    let edges = vec![(1u32, 0), (2, 0), (50, 1), (4, 2), (3, 0), (5, 50)];

    let max = edges.iter().fold(0, |m, (a, b)| m.max(*a).max(*b));
    assert!(max > (edges.len() * 2) as u32);

    let (encoded_edges, decoder) = encode_edges(&edges);
    let max = encoded_edges.iter().fold(0, |m, (a, b)| m.max(*a).max(*b));
    assert!(max < (edges.len() * 2) as u32);

    let mut decodable_edges = encoded_edges
        .into_iter()
        .map(|(a, b)| (a as usize, b as usize))
        .collect::<Vec<_>>();

    decode_edges(&mut decodable_edges, &decoder);

    assert_eq!(
        edges
            .into_iter()
            .map(|(a, b)| (a as usize, b as usize))
            .collect::<Vec<_>>(),
        decodable_edges
    );
}
