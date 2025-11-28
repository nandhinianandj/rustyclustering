//! Python bindings for ghdbscan clustering algorithms.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray2};
use ghdbscan_core::{
    DBSCAN as RustDBSCAN,
    HDBSCAN as RustHDBSCAN,
    DBSCANParams as RustDBSCANParams,
    HDBSCANParams as RustHDBSCANParams,
    DistanceMetric as RustDistanceMetric,
};

/// Distance metric options.
#[pyclass(eq, eq_int)]
#[derive(Clone, Debug, PartialEq)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
}

impl From<DistanceMetric> for RustDistanceMetric {
    fn from(metric: DistanceMetric) -> Self {
        match metric {
            DistanceMetric::Euclidean => RustDistanceMetric::Euclidean,
            DistanceMetric::Manhattan => RustDistanceMetric::Manhattan,
            DistanceMetric::Cosine => RustDistanceMetric::Cosine,
        }
    }
}

/// DBSCAN clustering algorithm.
///
/// Parameters
/// ----------
/// eps : float
///     Maximum distance between two samples for one to be considered as in the neighborhood of the other.
/// min_samples : int
///     The number of samples in a neighborhood for a point to be considered as a core point.
/// metric : str, optional
///     Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine'. Default is 'euclidean'.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from ghdbscan import DBSCAN
/// >>> X = np.array([[0, 0], [1, 1], [10, 10]])
/// >>> clustering = DBSCAN(eps=2.0, min_samples=2)
/// >>> labels = clustering.fit_predict(X)
#[pyclass]
pub struct DBSCAN {
    eps: f64,
    min_samples: usize,
    metric: DistanceMetric,
}

#[pymethods]
impl DBSCAN {
    #[new]
    #[pyo3(signature = (eps, min_samples, metric="euclidean"))]
    fn new(eps: f64, min_samples: usize, metric: &str) -> PyResult<Self> {
        let metric = match metric {
            "euclidean" => DistanceMetric::Euclidean,
            "manhattan" => DistanceMetric::Manhattan,
            "cosine" => DistanceMetric::Cosine,
            _ => return Err(PyValueError::new_err(format!("Unknown metric: {}", metric))),
        };

        Ok(Self {
            eps,
            min_samples,
            metric,
        })
    }

    /// Perform DBSCAN clustering on the data.
    ///
    /// Parameters
    /// ----------
    /// X : ndarray of shape (n_samples, n_features)
    ///     Training data.
    ///
    /// Returns
    /// -------
    /// labels : ndarray of shape (n_samples,)
    ///     Cluster labels for each point. Noisy samples are given the label -1.
    fn fit_predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let data = x.as_array();
        
        let params = RustDBSCANParams::new(self.eps, self.min_samples)
            .with_metric(self.metric.clone().into());
        
        let dbscan = RustDBSCAN::new(params);
        let result = dbscan.fit(&data);
        
        Ok(PyArray1::from_array_bound(py, &result.labels))
    }

    /// Get string representation.
    fn __repr__(&self) -> String {
        format!(
            "DBSCAN(eps={}, min_samples={}, metric={:?})",
            self.eps, self.min_samples, self.metric
        )
    }
}

/// HDBSCAN clustering algorithm.
///
/// Parameters
/// ----------
/// min_cluster_size : int
///     The minimum size of clusters.
/// min_samples : int, optional
///     The number of samples in a neighborhood for a point to be considered a core point.
///     Defaults to min_cluster_size if not specified.
/// metric : str, optional
///     Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine'. Default is 'euclidean'.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from ghdbscan import HDBSCAN
/// >>> X = np.array([[0, 0], [1, 1], [10, 10]])
/// >>> clustering = HDBSCAN(min_cluster_size=2)
/// >>> labels = clustering.fit_predict(X)
#[pyclass]
pub struct HDBSCAN {
    min_cluster_size: usize,
    min_samples: Option<usize>,
    metric: DistanceMetric,
}

#[pymethods]
impl HDBSCAN {
    #[new]
    #[pyo3(signature = (min_cluster_size, min_samples=None, metric="euclidean"))]
    fn new(min_cluster_size: usize, min_samples: Option<usize>, metric: &str) -> PyResult<Self> {
        let metric = match metric {
            "euclidean" => DistanceMetric::Euclidean,
            "manhattan" => DistanceMetric::Manhattan,
            "cosine" => DistanceMetric::Cosine,
            _ => return Err(PyValueError::new_err(format!("Unknown metric: {}", metric))),
        };

        Ok(Self {
            min_cluster_size,
            min_samples,
            metric,
        })
    }

    /// Perform HDBSCAN clustering on the data.
    ///
    /// Parameters
    /// ----------
    /// X : ndarray of shape (n_samples, n_features)
    ///     Training data.
    ///
    /// Returns
    /// -------
    /// labels : ndarray of shape (n_samples,)
    ///     Cluster labels for each point. Noisy samples are given the label -1.
    fn fit_predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let data = x.as_array();
        
        let params = RustHDBSCANParams::new(self.min_cluster_size, self.min_samples)
            .with_metric(self.metric.clone().into());
        
        let hdbscan = RustHDBSCAN::new(params);
        let result = hdbscan.fit(&data);
        
        Ok(PyArray1::from_array_bound(py, &result.labels))
    }

    /// Get string representation.
    fn __repr__(&self) -> String {
        format!(
            "HDBSCAN(min_cluster_size={}, min_samples={:?}, metric={:?})",
            self.min_cluster_size, self.min_samples, self.metric
        )
    }
}

/// Python module for DBSCAN and HDBSCAN clustering.
#[pymodule]
fn ghdbscan(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DBSCAN>()?;
    m.add_class::<HDBSCAN>()?;
    m.add_class::<DistanceMetric>()?;
    Ok(())
}
