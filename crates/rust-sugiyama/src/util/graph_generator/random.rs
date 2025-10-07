use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

pub fn gnm_graph_edges(n: usize, m: Option<usize>, seed: Option<u64>) -> Vec<(u32, u32)> {
    let m = m.unwrap_or(n - 1);
    let seed = seed.unwrap_or(42);
    let mut rng = StdRng::seed_from_u64(seed);

    let sources = (2..n as u32).collect::<Vec<_>>();
    let mut edges: HashSet<(u32, u32)> = [(1, 0)].into_iter().collect();
    let mut placed: HashSet<u32> = [0, 1].into_iter().collect();

    let mut i = 0;
    while edges.len() < m {
        let source = if i >= sources.len() {
            sources[rng.random_range(0..sources.len())]
        } else {
            sources[i]
        };

        let tix = rng.random_range(0..placed.len());
        let target = *placed.iter().collect::<Vec<_>>()[tix];
        if source == target || edges.contains(&(target, source)) {
            continue;
        }
        edges.insert((source, target));
        placed.insert(source);
        placed.insert(target);

        i += 1;
    }

    edges.into_iter().collect::<Vec<_>>()
}

#[test]
fn test_gnm_graph_edges() {
    println!("{:#?}", gnm_graph_edges(5, Some(7), None));
}
