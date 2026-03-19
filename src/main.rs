// use rand::Rng;
use std::time::Instant;
use rand::RngExt;
mod hnsw_manual;
use hnsw_manual::*;
fn main() {

    let graph = build_sample_graph();

    let query = vec![0.8, 0.85];

    let result = search(&graph, &query,20);

    println!("Search result node: {}", result);
    // let dataset_sizes = [10_000, 100_000];
    // let dimensions = [64, 128, 384];

    // for &n in &dataset_sizes {
    //     for &dim in &dimensions {

    //         println!("\n==============================");
    //         println!("Dataset size: {}", n);
    //         println!("Dimension: {}", dim);
    //         println!("==============================");

    //         let mut dataset = generate_embeddings(n, dim);

    //         let mut query = generate_embeddings(1, dim);

    //         normalize(&mut query[0..dim]);

    //         // normalize dataset
    //         for i in 0..n {
    //             let start = i * dim;
    //             let end = start + dim;
    //             normalize(&mut dataset[start..end]);
    //         }

    //         benchmark("Dot Product", &dataset, &query, n, dim, dot_product);
    //         benchmark("Cosine Similarity", &dataset, &query, n, dim, cosine_similarity);
    //         benchmark("Euclidean Distance", &dataset, &query, n, dim, euclidean_distance);
    //     }
    // }
}

fn benchmark(
    name: &str,
    dataset: &Vec<f32>,
    query: &[f32],
    n: usize,
    dim: usize,
    metric: fn(&[f32], &[f32]) -> f32,
) {

    let k = 10;

    let start_total = Instant::now();

    let (results, compute_time, sort_time) =
        brute_force_top_k(dataset, query, n, dim, k, metric);

    let total_time = start_total.elapsed();

    println!("\nMetric: {}", name);
    println!("Compute Time: {:?}", compute_time);
    println!("Sort Time: {:?}", sort_time);
    println!("Total Latency: {:?}", total_time);

    println!("Top {} results:", k);
    for (idx, score) in results {
        println!("Index: {}, Score: {}", idx, score);
    }
}

fn brute_force_top_k(
    dataset: &Vec<f32>,
    query: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    metric: fn(&[f32], &[f32]) -> f32,
) -> (Vec<(usize, f32)>, std::time::Duration, std::time::Duration) {

    let mut results: Vec<(usize, f32)> = Vec::with_capacity(n);

    let compute_start = Instant::now();

    for i in 0..n {

        let start = i * dim;
        let end = start + dim;

        let v = &dataset[start..end];

        let score = metric(query, v);

        results.push((i, score));
    }

    let compute_time = compute_start.elapsed();

    let sort_start = Instant::now();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    results.truncate(k);

    let sort_time = sort_start.elapsed();

    (results, compute_time, sort_time)
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