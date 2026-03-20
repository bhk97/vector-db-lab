use std::collections::{BinaryHeap, HashSet};
use std::cmp::Reverse;
use rand::RngExt;

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct OrderedF32(pub f32);

impl Eq for OrderedF32 {}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Clone)]
pub struct Node {
    id: usize,
    vector: Vec<f32>,
    level: usize,
    neighbors: Vec<Vec<usize>>, // neighbors[layer]
}

pub struct HNSW {
    nodes: Vec<Node>,
    entry_point: Option<usize>,
    max_level: usize,
    M: usize,
    ef_construction: usize,
}

impl HNSW {
    pub fn new(M: usize, ef_construction: usize) -> Self {
        Self {
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
            M,
            ef_construction,
        }
    }

    fn distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    fn calculate_ml(&self) -> f32 {
        1.0 / (self.M as f32).ln()
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::rng();
        let r: f32 = rng.random_range(f32::EPSILON..1.0);
        (-r.ln() * self.calculate_ml()) as usize
    }

    fn greedy_search_layer(
        &self,
        query: &[f32],
        mut current: usize,
        layer: usize,
    ) -> usize {
        loop {
            let mut improved = false;
            let current_dist = Self::distance(query, &self.nodes[current].vector);

            for &neighbor in &self.nodes[current].neighbors[layer] {
                let dist = Self::distance(query, &self.nodes[neighbor].vector);

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

    fn search_layer(
        &self,
        query: &[f32],
        entry: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<usize> {

        let mut visited = HashSet::new();

        let mut candidates: BinaryHeap<Reverse<(OrderedF32, usize)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedF32, usize)> = BinaryHeap::new();

        let entry_dist = OrderedF32(Self::distance(query, &self.nodes[entry].vector));

        candidates.push(Reverse((entry_dist, entry)));
        results.push((entry_dist, entry));
        visited.insert(entry);

        while let Some(Reverse((curr_dist, curr_node))) = candidates.pop() {

            let worst_dist = results.peek().unwrap().0;

            if curr_dist > worst_dist {
                break;
            }

            for &neighbor in &self.nodes[curr_node].neighbors[layer] {

                if visited.contains(&neighbor) {
                    continue;
                }

                visited.insert(neighbor);

                let dist = OrderedF32(Self::distance(query, &self.nodes[neighbor].vector));

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

    fn select_top_m(&self, candidates: &[usize]) -> Vec<usize> {
        candidates.iter().take(self.M).cloned().collect()
    }

    fn connect_bidirectional(
        &mut self,
        new_id: usize,
        neighbors: &[usize],
        layer: usize,
    ) {
        for &n in neighbors {
            self.nodes[new_id].neighbors[layer].push(n);
            self.nodes[n].neighbors[layer].push(new_id);

            // naive prune
            if self.nodes[n].neighbors[layer].len() > self.M {
                self.nodes[n].neighbors[layer].truncate(self.M);
            }
        }
    }

    pub fn insert(&mut self, vector: Vec<f32>) {

        let level = self.random_level();
        let new_id = self.nodes.len();

        self.nodes.push(Node {
            id: new_id,
            vector,
            level,
            neighbors: vec![vec![]; level + 1],
        });

        // first node
        if self.entry_point.is_none() {
            self.entry_point = Some(new_id);
            self.max_level = level;
            return;
        }

        let mut current = self.entry_point.unwrap();

        for l in (level + 1..=self.max_level).rev() {
            current = self.greedy_search_layer(
                &self.nodes[new_id].vector,
                current,
                l,
            );
        }

        for l in (0..=level.min(self.max_level)).rev() {

            let candidates = self.search_layer(
                &self.nodes[new_id].vector,
                current,
                self.ef_construction,
                l,
            );

            let neighbors = self.select_top_m(&candidates);

            self.connect_bidirectional(new_id, &neighbors, l);

            if let Some(&next) = neighbors.first() {
                current = next;
            }
        }

        // update entry point
        if level > self.max_level {
            self.entry_point = Some(new_id);
            self.max_level = level;
        }
    }

    pub fn print_graph(&self) {
        println!("\n=== HNSW Graph Visualization ===");
        println!("Max Level: {}, M: {}, ef_construction: {}", self.max_level, self.M, self.ef_construction);
        if let Some(ep) = self.entry_point {
            println!("Entry Point: Node {}", ep);
        }
        println!("================================\n");

        for l in (0..=self.max_level).rev() {
            println!("--- Layer {} ---", l);
            
            // Collect nodes at this level
            let mut level_nodes: Vec<usize> = self.nodes.iter()
                .enumerate()
                .filter(|(_, n)| n.level >= l)
                .map(|(i, _)| i)
                .collect();
            level_nodes.sort();

            for &id in &level_nodes {
                let node = &self.nodes[id];
                let neighbors = &node.neighbors[l];
                
                // Format neighbors string
                let neighbors_str = if neighbors.is_empty() {
                    "[]".to_string()
                } else {
                    format!("{:?}", neighbors)
                };

                // Visual representation of connections
                // [Node ID] --- (neighbors)
                print!("  [{:02}] ", id);
                if l < node.level {
                    print!("^ "); // Indicates it exists in higher layers
                } else {
                    print!("  ");
                }
                println!("connections: {}", neighbors_str);
            }
            println!("");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_visualization() {
        // Use a smaller M to make visualization cleaner
        let mut hnsw = HNSW::new(2, 4);
        
        // Insert some points
        // Using 2D points for simplicity
        let points = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![10.0, 10.0],
            vec![11.0, 11.0],
            vec![5.0, 5.0],
            vec![6.0, 6.0],
            vec![12.0, 12.0],
            vec![0.0, 0.0],
        ];

        for p in points {
            hnsw.insert(p);
        }

        hnsw.print_graph();
    }
}