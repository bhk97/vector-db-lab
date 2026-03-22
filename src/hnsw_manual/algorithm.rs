use std::collections::{BinaryHeap, HashSet};
use std::cmp::Reverse;
use rand::RngExt;
use crate::hnsw_manual::types::{Node, OrderedF32};
use crate::hnsw_manual::metrics::euclidean_distance;
use crate::hnsw_manual::HNSW;

pub fn calculate_ml(m: usize) -> f32 {
    1.0 / (m as f32).ln()
}

pub fn random_level(m: usize) -> usize {
    let mut rng = rand::rng();
    let r: f32 = rng.random_range(f32::EPSILON..1.0);
    (-r.ln() * calculate_ml(m)) as usize
}

pub fn greedy_search_layer(
    hnsw: &HNSW,
    query: &[f32],
    mut current: usize,
    layer: usize,
) -> usize {
    loop {
        let mut improved = false;
        let current_dist = euclidean_distance(query, &hnsw.nodes[current].vector);

        for &neighbor in &hnsw.nodes[current].neighbors[layer] {
            let dist = euclidean_distance(query, &hnsw.nodes[neighbor].vector);

            if dist < current_dist {
                current = neighbor;
                improved = true;
                break;
            }
        }

        if !improved {
            break;
        }
    }
    current
}

pub fn search_layer(
    hnsw: &HNSW,
    query: &[f32],
    entry: usize,
    ef: usize,
    layer: usize,
) -> Vec<usize> {
    let mut visited = HashSet::new();
    let mut candidates: BinaryHeap<Reverse<(OrderedF32, usize)>> = BinaryHeap::new();
    let mut results: BinaryHeap<(OrderedF32, usize)> = BinaryHeap::new();

    let entry_dist = OrderedF32(euclidean_distance(query, &hnsw.nodes[entry].vector));

    candidates.push(Reverse((entry_dist, entry)));
    results.push((entry_dist, entry));
    visited.insert(entry);

    while let Some(Reverse((curr_dist, curr_node))) = candidates.pop() {
        let worst_dist = results.peek().unwrap().0;

        if curr_dist > worst_dist {
            break;
        }

        for &neighbor in &hnsw.nodes[curr_node].neighbors[layer] {
            if visited.contains(&neighbor) {
                continue;
            }

            visited.insert(neighbor);
            let dist = OrderedF32(euclidean_distance(query, &hnsw.nodes[neighbor].vector));

            if results.len() < ef || dist < results.peek().unwrap().0 {
                candidates.push(Reverse((dist, neighbor)));
                results.push((dist, neighbor));

                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }

    let mut res: Vec<_> = results.into_iter().collect();
    res.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    res.into_iter().map(|(_, id)| id).collect()
}

pub fn select_top_m(candidates: &[usize], m: usize) -> Vec<usize> {
    candidates.iter().take(m).cloned().collect()
}

pub fn connect_bidirectional(
    hnsw: &mut HNSW,
    new_id: usize,
    neighbors: &[usize],
    layer: usize,
) {
    let m = hnsw.m;
    for &n in neighbors {
        hnsw.nodes[new_id].neighbors[layer].push(n);
        hnsw.nodes[n].neighbors[layer].push(new_id);

        // naive prune
        if hnsw.nodes[n].neighbors[layer].len() > m {
            hnsw.nodes[n].neighbors[layer].truncate(m);
        }
    }
}

pub fn insert(hnsw: &mut HNSW, vector: Vec<f32>) {
    let level = random_level(hnsw.m);
    let new_id = hnsw.nodes.len();

    hnsw.nodes.push(Node {
        id: new_id,
        vector,
        level,
        neighbors: vec![vec![]; level + 1],
    });

    // first node
    if hnsw.entry_point.is_none() {
        hnsw.entry_point = Some(new_id);
        hnsw.max_level = level;
        return;
    }

    let mut current = hnsw.entry_point.unwrap();

    // Sinking down from max_level to level + 1
    for l in (level + 1..=hnsw.max_level).rev() {
        current = greedy_search_layer(hnsw, &hnsw.nodes[new_id].vector, current, l);
    }

    // Creating connections from level down to 0
    for l in (0..=level.min(hnsw.max_level)).rev() {
        let candidates = search_layer(
            hnsw,
            &hnsw.nodes[new_id].vector,
            current,
            hnsw.ef_construction,
            l,
        );

        let neighbors = select_top_m(&candidates, hnsw.m);
        connect_bidirectional(hnsw, new_id, &neighbors, l);

        if let Some(&next) = neighbors.first() {
            current = next;
        }
    }

    // update entry point
    if level > hnsw.max_level {
        hnsw.entry_point = Some(new_id);
        hnsw.max_level = level;
    }
}
