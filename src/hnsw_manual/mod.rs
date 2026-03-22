pub mod types;
pub mod metrics;
pub mod algorithm;
pub mod visualize;

pub use types::Node;

pub struct HNSW {
    pub nodes: Vec<Node>,
    pub entry_point: Option<usize>,
    pub max_level: usize,
    pub m: usize,
    pub ef_construction: usize,
}

impl HNSW {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
            m,
            ef_construction,
        }
    }

    pub fn insert(&mut self, vector: Vec<f32>) {
        algorithm::insert(self, vector);
    }

    pub fn print_graph(&self) {
        visualize::print_graph(self);
    }
}

#[cfg(test)]
mod tests;
