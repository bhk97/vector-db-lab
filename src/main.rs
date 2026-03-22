use hnsw_manual::HNSW;

mod hnsw_manual;

fn main() {
    // 1. Initialize HNSW
    // M=4, ef_construction=20 for a small sample
    let mut hnsw = HNSW::new(4, 20);

    // 2. Generate and Insert Sample Vectors
    let num_points = 10;
    let dim = 2; // 2D vectors for easier mental visualization
    
    println!("Inserting {} points into HNSW...", num_points);
    
    for i in 0..num_points {
        let vector: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
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