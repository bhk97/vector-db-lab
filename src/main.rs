use std::time::Instant;
use rand::RngExt;

mod hnsw_manual;
use hnsw_manual::*;

fn main() {
    // 1. Initialize HNSW
    // M=4, ef_construction=20 for a small sample
    let mut hnsw = HNSW::new(4, 20);

    // 2. Generate and Insert Sample Vectors
    let num_points = 10;
    let dim = 2; // 2D vectors for easier mental visualization
    
    println!("Inserting {} points into HNSW...", num_points);
    
    for i in 0..num_points {
        let mut rng = rand::rng();
        let vector: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        hnsw.insert(vector);
        println!("  Inserted Node {}", i);
    }

    // 3. Visualize the Graph
    hnsw.print_graph();

    // Optional: Benchmark and results if needed (commented out previous code)
    /*
    let query = vec![0.5, 0.5];
    // Search logic could be added here if implemented in HNSW
    */
}

fn generate_embeddings(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    let mut data = Vec::with_capacity(n * dim);
    for _ in 0..n * dim {
        data.push(rng.random::<f32>());
    }
    data
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum.sqrt()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

fn normalize(v: &mut [f32]) {
    let mut sum = 0.0;
    for x in v.iter() {
        sum += x * x;
    }
    let norm = sum.sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}