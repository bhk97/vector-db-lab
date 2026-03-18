pub struct Node {
    id: usize,
    vector: Vec<f32>,
    neighbors: Vec<Vec<usize>>,
}

pub struct HNSW {
    nodes: Vec<Node>,
    max_level:usize,
    entry_point:usize
}

pub fn build_sample_graph() -> HNSW {

    let a = Node {
        id: 0,
        vector: vec![0.1, 0.2],
        neighbors: vec![vec![], vec![], vec![]],
    };

    let b = Node {
        id: 1,
        vector: vec![0.3, 0.2],
        neighbors: vec![vec![], vec![]],
    };

    let c = Node {
        id: 2,
        vector: vec![0.5, 0.6],
        neighbors: vec![vec![]],
    };

    let d = Node {
        id: 3,
        vector: vec![0.7, 0.8],
        neighbors: vec![vec![]],
    };

    let e = Node {
        id: 4,
        vector: vec![0.9, 0.9],
        neighbors: vec![vec![]],
    };

    let mut graph = HNSW {
        nodes: vec![a, b, c, d, e],
        entry_point: 0,
        max_level: 2,
    };

    graph.nodes[0].neighbors[0] = vec![1];
    graph.nodes[1].neighbors[0] = vec![0,2];
    graph.nodes[2].neighbors[0] = vec![1,3];
    graph.nodes[3].neighbors[0] = vec![2,4];
    graph.nodes[4].neighbors[0] = vec![3];

    graph.nodes[0].neighbors[1] = vec![1];
    graph.nodes[1].neighbors[1] = vec![0];

    graph
}

pub fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

pub fn search(graph: &HNSW, query: &[f32]) -> usize {

    let mut current = graph.entry_point;

    println!("Starting search from node {}", current);

    for layer in (0..=graph.max_level).rev() {

        println!("--- Searching Layer {} ---", layer);

        loop {

            let mut improved = false;

            let current_dist = distance(query, &graph.nodes[current].vector);

            println!(
                "Current Node: {} Distance: {}",
                current,
                current_dist
            );

            for &neighbor in &graph.nodes[current].neighbors[layer] {

                let neighbor_dist = distance(query, &graph.nodes[neighbor].vector);

                println!(
                    "Checking neighbor {} distance {}",
                    neighbor,
                    neighbor_dist
                );

                if neighbor_dist < current_dist {

                    println!(
                        "Moving from node {} → {}",
                        current,
                        neighbor
                    );

                    current = neighbor;
                    improved = true;

                    break;
                }
            }

            if !improved {

                println!(
                    "No better neighbor found at layer {}",
                    layer
                );

                break;
            }
        }
    }

    println!("Final closest node: {}", current);

    current
}
