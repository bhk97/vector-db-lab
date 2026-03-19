
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Reverse;

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct OrderedF32(pub f32);

impl Eq for OrderedF32 {}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

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

pub fn search(graph: &HNSW, query: &[f32], ef: usize) -> usize {

    let mut current = graph.entry_point;

    println!("Starting search from node {}", current);

    
    for layer in (1..=graph.max_level).rev() {

        println!("--- Greedy Layer {} ---", layer);

        loop {
            let mut improved = false;

            let current_dist = distance(query, &graph.nodes[current].vector);

            for &neighbor in &graph.nodes[current].neighbors[layer] {

                let neighbor_dist = distance(query, &graph.nodes[neighbor].vector);

                if neighbor_dist < current_dist {
                    println!("Layer {}: {} → {}", layer, current, neighbor);

                    current = neighbor;
                    improved = true;
                    break;
                }
            }

            if !improved {
                break;
            }
        }
    }

    

    println!("--- efSearch Layer 0 ---");

    let mut visited = HashSet::new();

    // candidates → min-heap (closest first)
    let mut candidates: BinaryHeap<Reverse<(OrderedF32, usize)>> = BinaryHeap::new();

    // results → max-heap (worst at top)
    let mut results: BinaryHeap<(OrderedF32, usize)> = BinaryHeap::new();

    let entry_dist = OrderedF32(distance(query, &graph.nodes[current].vector));

    candidates.push(Reverse((entry_dist, current)));
    results.push((entry_dist, current));
    visited.insert(current);

    while let Some(Reverse((curr_dist, curr_node))) = candidates.pop() {

        let worst_dist = results.peek().unwrap().0;

        
        if curr_dist > worst_dist {
            break;
        }

        for &neighbor in &graph.nodes[curr_node].neighbors[0] {

            if visited.contains(&neighbor) {
                continue;
            }

            visited.insert(neighbor);

            let dist = OrderedF32(distance(query, &graph.nodes[neighbor].vector));

            if results.len() < ef || dist < results.peek().unwrap().0 {

                println!("Exploring {} → {}", curr_node, neighbor);

                candidates.push(Reverse((dist, neighbor)));
                results.push((dist, neighbor));

                if results.len() > ef {
                    results.pop(); // remove worst
                }
            }
        }
    }

    
    let best = results
        .into_iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap();

    println!("Final closest node: {}", best.1);

    best.1
}