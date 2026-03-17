use rand::Rng;
use std::time::Instant;
use rand::RngExt;
fn main() {

    // experiments
    let dataset_sizes = [10_000, 100_000];
    let dimensions = [64, 128, 384];

    for &n in &dataset_sizes {
        for &dim in &dimensions {

            println!("\n==============================");
            println!("Dataset size: {}", n);
            println!("Dimension: {}", dim);
            println!("==============================");

            let dataset = generate_embeddings(n, dim);

            let mut query = generate_embeddings(1, dim).remove(0);

            normalize(&mut query);

            // normalize dataset
            let mut dataset = dataset;
            for v in dataset.iter_mut() {
                normalize(v);
            }

            benchmark("Dot Product", &dataset, &query, dot_product);
            benchmark("Cosine Similarity", &dataset, &query, cosine_similarity);
            benchmark("Euclidean Distance", &dataset, &query, euclidean_distance);
        }
    }
}

fn benchmark(
    name: &str,
    dataset: &Vec<Vec<f32>>,
    query: &[f32],
    metric: fn(&[f32], &[f32]) -> f32,
) {

    let k = 10;

    let start = Instant::now();

    let results = brute_force_top_k(dataset, query, k, metric);

    let duration = start.elapsed();

    println!("\nMetric: {}", name);
    println!("Latency: {:?}", duration);

    println!("Top {} results:", k);
    for (idx, score) in results {
        println!("Index: {}, Score: {}", idx, score);
    }
}

fn brute_force_top_k(
    dataset: &Vec<Vec<f32>>,
    query: &[f32],
    k: usize,
    metric: fn(&[f32], &[f32]) -> f32,
) -> Vec<(usize, f32)> {

    let mut results: Vec<(usize, f32)> = Vec::with_capacity(dataset.len());

    for (i, v) in dataset.iter().enumerate() {
        let score = metric(query, v);
        results.push((i, score));
    }

    // partial sort for top-k
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    results.truncate(k);

    results
}

fn generate_embeddings(n: usize, dim: usize) -> Vec<Vec<f32>> {

    let mut rng = rand::rng();

    let mut data = Vec::with_capacity(n);

    for _ in 0..n {
        let mut v = Vec::with_capacity(dim);

        for _ in 0..dim {
            v.push(rng.random::<f32>());
        }

        data.push(v);
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