/// Author: paddison
/// Repository: https://github.com/paddison/graph_generator/blob/72ae1b0/src/layered.rs
/// Date Cited: 2025/10/04
///
/// Creates a graph with `num_nodes` vertices, which have `edges_per_node` edges.
/// The layout of the graph is layered, where it grows from one vertice to a certain maximum with
/// and then starts shrinking again to one vertice at the bottom layer:
/// ```ignore
///             /---v---\
///          /-v-\    /-v-\
///         v    v   v    v
///          \-v-/   \-v-/
///            \---v---/
/// ```
/// The number of layers depends on the number of vertices.
/// More specifically, if edges^(n) - 1 + edges^(n - 1) <= num_vertices <= edges^(n+1) - 1 + edges^(n),
/// we will have at maximum 2n + 1 layers, and at minimum n layers.
#[derive(Debug, Eq, PartialEq)]
pub struct LayeredGraph {
    growing_layers: u32,
    shrinking_layers: u32,
    num_nodes: u32,
    edges_per_node: u32,
}

impl LayeredGraph {
    pub fn new(
        growing_nodes: u32,
        shrinking_nodes: u32,
        num_nodes: u32,
        edges_per_node: u32,
    ) -> Self {
        LayeredGraph {
            growing_layers: growing_nodes,
            shrinking_layers: shrinking_nodes,
            num_nodes,
            edges_per_node,
        }
    }

    /// Creates a new layout which will have `num_nodes` vertices and `edges_per_node` edges per vertice.
    pub fn new_from_num_nodes(num_nodes: u32, edges_per_node: u32) -> Self {
        let calc_num_nodes =
            |edges_per_node: u32, pow: u32| (edges_per_node.pow(pow) - 1) / (edges_per_node - 1);
        for i in 0.. {
            let growing_nodes = calc_num_nodes(edges_per_node, i);
            let shrinking_nodes = calc_num_nodes(edges_per_node, i + 1);
            if growing_nodes + shrinking_nodes >= num_nodes {
                return LayeredGraph {
                    growing_layers: i,
                    shrinking_layers: (i + 1),
                    num_nodes,
                    edges_per_node,
                };
            }
        }
        unreachable!()
    }

    /// Build the edges of the graph.
    pub fn build_edges(&self) -> Vec<(u32, u32)> {
        // start with node = 0
        let mut edges = Vec::new();
        let mut node = 0;
        for layer in 0..self.growing_layers {
            let layer_size = self.edges_per_node.pow(layer);
            for _ in 0..layer_size {
                for edge in 1..=self.edges_per_node {
                    edges.push((node, self.edges_per_node * node + edge));
                    if edges.len() as u32 + 1 == self.num_nodes {
                        return edges;
                    }
                }
                node += 1;
            }
        }

        for layer in (1..self.shrinking_layers).rev() {
            let mut layer_size = self.edges_per_node.pow(layer);
            for _ in 0..(layer_size / self.edges_per_node) {
                for edge in 0..self.edges_per_node {
                    let successor = node + layer_size - edge;
                    if successor >= self.num_nodes {
                        return edges;
                    }
                    edges.push((node, node + layer_size - edge));
                    node += 1;
                }
                layer_size -= self.edges_per_node - 1;
            }
        }
        edges
    }
}

#[cfg(test)]
mod tests {
    use super::LayeredGraph;

    #[test]
    fn test_new_from_num_nodes_2_nodes_2_edges() {
        let expected = LayeredGraph::new(1, 2, 2, 2);
        let actual = LayeredGraph::new_from_num_nodes(2, 2);
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_new_from_num_nodes_8_nodes_2_edges() {
        let expected = LayeredGraph::new(2, 3, 8, 2);
        let actual = LayeredGraph::new_from_num_nodes(8, 2);
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_new_from_num_nodes_22_nodes_2_edges() {
        let expected = LayeredGraph::new(3, 4, 22, 2);
        let actual = LayeredGraph::new_from_num_nodes(22, 2);
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_new_from_num_nodes_16_nodes_3_edges() {
        let expected = LayeredGraph::new(2, 3, 16, 3);
        let actual = LayeredGraph::new_from_num_nodes(16, 3);
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_print_edges() {
        let layout = LayeredGraph::new_from_num_nodes(766, 2);
        println!("{:?}", layout);
        println!("{:?}", layout.build_edges());
    }
}
