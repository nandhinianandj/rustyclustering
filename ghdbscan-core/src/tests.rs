#[cfg(test)]
mod tests {
    use crate::*;
    use ndarray::array;

    #[test]
    fn test_integration_dbscan() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [10.0, 10.0],
            [11.0, 10.0],
            [10.0, 11.0],
        ];
        
        let params = DBSCANParams::new(2.0, 2);
        let dbscan = DBSCAN::new(params);
        let result = dbscan.fit(&data.view());
        
        assert_eq!(result.n_clusters(), 2);
        println!("DBSCAN found {} clusters", result.n_clusters());
    }

    #[test]
    fn test_integration_hdbscan() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [10.0, 10.0],
            [11.0, 10.0],
            [10.0, 11.0],
        ];
        
        let params = HDBSCANParams::new(2, None);
        let hdbscan = HDBSCAN::new(params);
        let result = hdbscan.fit(&data.view());
        
        assert!(result.n_clusters() >= 1);
        println!("HDBSCAN found {} clusters", result.n_clusters());
    }
}
