use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rust_sugiyama::configure::{Config, CrossingMinimization};
use rust_sugiyama::graph_generator::{LayeredGraph, gnm_graph_edges};

type EdgeFn = fn(u32) -> Vec<(u32, u32)>;
type NamedEdgeFn = (&'static str, EdgeFn);

const TOPOLOGIES: &[NamedEdgeFn] = &[
    // edges_per_node = 2 gives a binary-tree shaped layered graph
    ("layered", |n| {
        LayeredGraph::new_from_num_nodes(n, 2).build_edges()
    }),
    // using m = None and a seed value gives a reproducible random DAG
    ("random", |n| gnm_graph_edges(n as usize, None, Some(42))),
];

/// Concatenate `n_components` independent copies of a graph by shifting node
/// IDs, producing a graph with exactly `n_components` weakly-connected
/// components.  `edge_fn` is called once per component to generate the base
/// edge list.
fn make_components(
    n_nodes: u32,
    n_components: u32,
    edge_fn: impl Fn(u32) -> Vec<(u32, u32)>,
) -> Vec<(u32, u32)> {
    (0..n_components)
        .flat_map(|c| {
            let offset = c * n_nodes;
            edge_fn(n_nodes)
                .into_iter()
                .map(move |(s, t)| (s + offset, t + offset))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_config() -> Config {
    Config {
        check_layout: false,
        ..Config::default()
    }
}

// ---------------------------------------------------------------------------
// Benchmark groups
// ---------------------------------------------------------------------------

/// Scale total graph size
fn bench_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("size_scaling");

    for (name, func) in TOPOLOGIES {
        for &n in &[100u32, 500, 1000, 5000] {
            let config = default_config();

            let edges = func(n);
            group.bench_with_input(
                BenchmarkId::new(name.to_string(), n),
                &(edges, config),
                |b, (e, cfg)| {
                    b.iter(|| rust_sugiyama::from_edges(e, cfg));
                },
            );
        }
    }

    group.finish();
}

/// Vary number of weakly-connected components to isolate the parallelism benefit across components.
fn bench_component_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("component_count");

    // (n_nodes, n_components) pairs — total ≈ 1024 nodes each
    let combos = &[(1024u32, 1), (512, 2), (256, 4), (128, 8), (64, 16)];
    for (name, func) in TOPOLOGIES {
        for &(n_nodes, n_components) in combos {
            let edges = make_components(n_nodes, n_components, *func);
            let config = default_config();
            group.bench_with_input(
                BenchmarkId::new(name.to_string(), format!("{n_nodes}x{n_components}")),
                &(edges, config),
                |b, (e, cfg)| {
                    b.iter(|| rust_sugiyama::from_edges(e, cfg));
                },
            );
        }
    }

    group.finish();
}

/// Isolate the crossing-minimization strategy and the transpose step.
fn bench_crossing_minimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossing_minimization");

    let strategies: &[(&str, CrossingMinimization, bool)] = &[
        (
            "barycenter+transpose",
            CrossingMinimization::Barycenter,
            true,
        ),
        ("barycenter_only", CrossingMinimization::Barycenter, false),
        ("median+transpose", CrossingMinimization::Median, true),
        ("median_only", CrossingMinimization::Median, false),
    ];

    for (name, func) in TOPOLOGIES {
        for (strategy_name, cm, transpose) in strategies {
            let config = Config {
                c_minimization: *cm,
                transpose: *transpose,
                check_layout: false,
                ..Config::default()
            };
            let edges = func(1000);
            group.bench_with_input(
                BenchmarkId::new(name.to_string(), strategy_name),
                &(edges.clone(), config),
                |b, (e, cfg)| {
                    b.iter(|| rust_sugiyama::from_edges(e, cfg));
                },
            );
        }
    }

    group.finish();
}

/// Large graphs — fewer samples to keep total bench time reasonable.
fn bench_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("large");
    group.sample_size(20);

    // (n_nodes, n_components) — total nodes stays 10,000 each
    let combos = &[(10000, 1), (1000, 10), (500, 20)];

    for (name, edge_fn) in TOPOLOGIES {
        for &(n_nodes, n_components) in combos {
            let config = default_config();
            let edges = make_components(n_nodes, n_components, *edge_fn);
            let param = format!("{n_nodes}n_{n_components}x");
            group.bench_with_input(
                BenchmarkId::new(name.to_string(), param),
                &(edges, config),
                |b, (e, cfg)| {
                    b.iter(|| rust_sugiyama::from_edges(e, cfg));
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_size_scaling,
    bench_component_count,
    bench_crossing_minimization,
    bench_large,
);
criterion_main!(benches);
