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
    pub id: usize,
    pub vector: Vec<f32>,
    pub level: usize,
    pub neighbors: Vec<Vec<usize>>, // neighbors[layer]
}
