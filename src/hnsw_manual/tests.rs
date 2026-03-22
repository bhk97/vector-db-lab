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
