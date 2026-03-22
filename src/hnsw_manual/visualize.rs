use crate::hnsw_manual::HNSW;

pub fn print_graph(hnsw: &HNSW) {
    println!("\n=== HNSW Graph Layered Visualization ===");
    let n = hnsw.nodes.len();
    if n == 0 {
        println!("(Graph is empty)");
        return;
    }

    // Each node occupies a fixed column to show "sink" through layers
    let col_width = 8;

    for l in (0..=hnsw.max_level).rev() {
        println!("\n--- LAYER {} ---", l);

        // 1. Print node markers
        let mut node_line = String::new();
        let mut conn_line = String::new();

        for i in 0..n {
            if hnsw.nodes[i].level >= l {
                node_line.push_str(&format!("[{:02}]", i));

                // Show connections summary below the node
                let neighbors = &hnsw.nodes[i].neighbors[l];
                if !neighbors.is_empty() {
                    conn_line.push_str(&format!(" {:>3} ", format!("n:{}", neighbors.len())));
                } else {
                    conn_line.push_str("     ");
                }
            } else {
                node_line.push_str(" .. ");
                conn_line.push_str("     ");
            }
            node_line.push_str(&" ".repeat(col_width - 4));
            conn_line.push_str(&" ".repeat(col_width - 5));
        }
        println!("{}", node_line);
        println!("{}", conn_line);

        // 2. Draw vertical connectors to next layer
        if l > 0 {
            let mut vert_line = String::new();
            for i in 0..n {
                if hnsw.nodes[i].level >= l {
                    vert_line.push_str("  | ");
                } else {
                    vert_line.push_str("    ");
                }
                vert_line.push_str(&" ".repeat(col_width - 4));
            }
            println!("{}", vert_line);
        }
    }

    println!("\nLegend: [XX] = Node exists in layer, .. = Node starts in lower layer, n:Y = Node has Y neighbors in this layer");

    println!("\nDetailed Edges per Layer:");
    for l in (0..=hnsw.max_level).rev() {
        print!("L{}: ", l);
        let mut layer_edges = Vec::new();
        for (i, node) in hnsw.nodes.iter().enumerate() {
            if node.level >= l && !node.neighbors[l].is_empty() {
                layer_edges.push(format!("({} -> {:?})", i, node.neighbors[l]));
            }
        }
        println!("{}", layer_edges.join(", "));
    }
    println!("========================================\n");
}
