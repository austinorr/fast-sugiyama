use petgraph::{
    Direction,
    stable_graph::{NodeIndex, StableDiGraph},
    visit::EdgeRef,
};

use svg::{
    Document,
    node::{
        Comment,
        element::{Definitions, Line, Marker, Path, Rectangle, Text},
    },
};

use rust_sugiyama::{Edge, Vertex, configure::Config, from_edges};

/// Generates an SVG string of the resulting layout. This is an example of
/// rendering the layouts.
fn create_svg<E>(
    graph: &StableDiGraph<Vertex, E>,
    layout: &[(usize, (f64, f64))],
    character_width: f64,
    node_height: f64,
    padding: (f64, f64),
) -> String {
    // Figure out the extents of the SVG.
    let node_size = |idx| {
        (
            character_width
                + graph[NodeIndex::new(idx)].id.to_string().len() as f64 / 2.0 * character_width,
            node_height,
        )
    };

    let (min_x, max_x, min_y, max_y) = layout
        .iter()
        .map(|(idx, (x, y))| {
            let (w, h) = node_size(*idx);
            (*x - w * 0.5, *x + w * 0.5, *y - h * 0.5, *y + h * 0.5)
        })
        .fold(
            (
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ),
            |acc, (x1, x2, y1, y2)| (acc.0.min(x1), acc.1.max(x2), acc.2.min(y1), acc.3.max(y2)),
        );

    let mut document = Document::new()
        .set("width", max_x - min_x + 2.0 * padding.0)
        .set("height", max_y - min_y + 2.0 * padding.1);

    let origin = (padding.0 - min_x, padding.1 - min_y);

    // Generate a line for each edge in the graph.
    document = document.add(
        Definitions::new().add(
            Marker::new()
                .set("id", "head")
                .set("orient", "auto")
                .set("markerWidth", "25")
                .set("markerHeight", "10")
                .set("refX", (20.0 + node_height / 2.0).to_string())
                .set("refY", "5")
                .add(
                    Path::new()
                        .set("d", "M 0 0 L 25 5 L 0 10 z")
                        .set("fill", "black"),
                ),
        ),
    );
    for (id, pos) in layout {
        let idx = NodeIndex::new(*id);
        let vertex_name = graph[idx].id.to_string();
        for e in graph.edges_directed(idx, Direction::Outgoing) {
            let other_idx = e.target();
            let other_vertex = graph[other_idx];
            let other_vertex_name = other_vertex.id.to_string();
            let (_, other_vertex_pos) = layout
                .iter()
                .find(|(ix, _)| *ix == other_idx.index())
                .unwrap();

            let mut line = Line::new()
                .set("x1", pos.0 + origin.0)
                .set("y1", max_y - pos.1 + origin.1)
                .set("x2", other_vertex_pos.0 + origin.0)
                .set("y2", max_y - other_vertex_pos.1 + origin.1)
                .set("style", "stroke:black;")
                .add(Comment::new(format!(
                    "({vertex_name}, {other_vertex_name})"
                )));
            if !other_vertex.is_dummy {
                line = line.set("marker-end", "url(#head)");
            }
            document = document.add(line);
        }
    }

    // Generate a rectangle for each vertex in the graph.
    for (id, pos) in layout {
        let vertex = graph[NodeIndex::new(*id)];
        if vertex.is_dummy {
            continue;
        }
        let vertex_name = vertex.id.to_string();
        let size = (
            character_width + vertex_name.len() as f64 / 2.0 * character_width,
            node_height,
        );

        let rect = Rectangle::new()
            .set("x", pos.0 - size.0 * 0.5 + origin.0)
            .set("y", max_y - pos.1 - size.1 * 0.5 + origin.1)
            .set("width", size.0)
            .set("height", size.1)
            .set("rx", node_height / 4.0)
            .set("ry", node_height / 6.0)
            .set("style", "fill:white;stroke:black;");

        let text = Text::new(vertex_name)
            .set("x", pos.0 + origin.0)
            .set("y", max_y - pos.1 + origin.1)
            .set("dominant-baseline", "middle")
            .set("text-anchor", "middle");
        document = document.add(rect).add(text);
    }

    document.to_string()
}

fn main() {
    env_logger::init();

    let edges = vec![
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
        //
        (51, 50),
        (52, 51),
        (53, 50),
        (54, 51),
        (55, 50),
        (54, 50),
    ];

    let vertex_spacing = 72.0;

    let layouts = from_edges(
        &edges,
        &Config {
            vertex_spacing,
            ..Default::default()
        },
    );

    let nodes = edges
        .iter()
        .fold(std::collections::HashSet::new(), |mut acc, p| {
            acc.insert(p.0);
            acc.insert(p.1);
            acc
        });

    for (i, (layout, _w, _h, el)) in layouts.iter().enumerate() {
        let graph = match el {
            Some(edge_list) => {
                &StableDiGraph::from_edges(edge_list.iter().map(|(s, t)| (*s as u32, *t as u32)))
                    .map(
                        |id, _: &Vertex| {
                            let mut v = Vertex::new(id.index(), (0.0, 0.0));
                            v.is_dummy = !nodes.contains(&(v.id as u32));
                            v
                        },
                        |_, _: &Edge| Edge::default(),
                    )
            }
            None => &StableDiGraph::from_edges(&edges).map(
                |id, _: &Vertex| Vertex::new(id.index(), (0.0, 0.0)),
                |_, _e: &Edge| Edge::default(),
            ),
        };

        let svg = create_svg(graph, layout, 20.0, 25.0, (vertex_spacing, vertex_spacing));

        let fname = format!("./misc/graph_{i}.svg");
        let mut file = std::fs::File::create(&fname).expect("Unable to create svg file");
        use std::io::Write;
        // Write the content (as bytes) to the file
        file.write_all(svg.as_bytes())
            .expect("Unable to write data");
    }

    println!("complete");
}
