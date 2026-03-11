mod layered;
pub use layered::LayeredGraph;

#[cfg(feature = "rand")]
mod random;
#[cfg(feature = "rand")]
pub use random::gnm_graph_edges;
