use log::LevelFilter;

use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3_log::{Caching, Logger};

use rust_sugiyama::{
    Layouts,
    configure::{Config, CrossingMinimization, RankingType},
    from_edges,
};

#[pymodule]
fn _fast_sugiyama(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    let handle = Logger::new(_py, Caching::LoggersAndLevels)?
        .filter(LevelFilter::Trace)
        .install()
        .expect("Someone installed a logger before us :-(");

    handle.reset();

    m.add_wrapped(wrap_pyfunction!(from_edges_py))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "from_edges")]
fn from_edges_py(
    edges: &Bound<'_, PyList>,
    minimum_length: Option<u32>,
    vertex_spacing: Option<f64>,
    dummy_vertices: Option<bool>,
    dummy_size: Option<f64>,
    ranking_type: Option<String>,
    crossing_minimization: Option<String>,
    transpose: Option<bool>,
    check_layout: Option<bool>,
    encode_edges: Option<bool>,
) -> PyResult<Layouts<usize>> {
    let mut _edges: Vec<(u32, u32)> = edges.extract()?;

    let mut config = Config::default();

    if let Some(minimum_length) = minimum_length {
        config.minimum_length = minimum_length;
    }

    if let Some(vertex_spacing) = vertex_spacing {
        config.vertex_spacing = vertex_spacing;
    }

    if let Some(dummy_vertices) = dummy_vertices {
        config.dummy_vertices = dummy_vertices;
    }

    if let Some(dummy_size) = dummy_size {
        config.dummy_size = dummy_size;
    }

    if let Some(ranking_type) = ranking_type {
        config.ranking_type =
            RankingType::try_from(ranking_type.to_lowercase()).expect("invalid ranking type");
    }

    if let Some(crossing_minimization) = crossing_minimization {
        config.c_minimization =
            CrossingMinimization::try_from(crossing_minimization.to_lowercase())
                .expect("invalid crossing minimization");
    }

    if let Some(transpose) = transpose {
        config.transpose = transpose;
    }

    if let Some(check_layout) = check_layout {
        config.check_layout = check_layout;
    }

    if let Some(encode_edges) = encode_edges {
        config.encode_edges = encode_edges;
    }

    let layouts = from_edges(&_edges, &config);

    Ok(layouts)
}
