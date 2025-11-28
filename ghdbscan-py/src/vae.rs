use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray1, PyArray2, ToPyArray};
use candle_core::{Device, Tensor};
use ghdbscan_vae::{VaDE, VaDEConfig, TrainingConfig, VaDETrainer};

#[pyclass]
pub struct VaDE_Clustering {
    trainer: Option<VaDETrainer>,
    config: VaDEConfig,
    training_config: TrainingConfig,
    device: Device,
}

#[pymethods]
impl VaDE_Clustering {
    #[new]
    #[pyo3(signature = (input_dim, latent_dim, n_clusters, epochs=100, batch_size=32, learning_rate=0.0001, use_gpu=true))]
    fn new(
        input_dim: usize,
        latent_dim: usize,
        n_clusters: usize,
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        use_gpu: bool,
    ) -> PyResult<Self> {
        let device = if use_gpu && candle_core::utils::cuda_is_available() {
            Device::new_cuda(0).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        } else {
            Device::Cpu
        };

        let config = VaDEConfig::new(input_dim, latent_dim, n_clusters);
        let training_config = TrainingConfig::new()
            .with_epochs(epochs)
            .with_batch_size(batch_size)
            .with_learning_rate(learning_rate);

        Ok(Self {
            trainer: None,
            config,
            training_config,
            device,
        })
    }

    fn fit(&mut self, data: PyReadonlyArray2<f32>) -> PyResult<Vec<f32>> {
        let array = data.as_array();
        let shape = array.shape();
        let n_samples = shape[0];
        let n_features = shape[1];

        if n_features != self.config.input_dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Input dimension mismatch: expected {}, got {}", self.config.input_dim, n_features)
            ));
        }

        // Convert to Candle tensor
        let flat_data: Vec<f32> = array.iter().cloned().collect();
        let tensor = Tensor::from_vec(flat_data, (n_samples, n_features), &self.device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Initialize trainer if not already done
        if self.trainer.is_none() {
            self.trainer = Some(
                VaDETrainer::new(self.config.clone(), self.training_config.clone(), &self.device)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            );
        }

        // Train
        if let Some(trainer) = &mut self.trainer {
            let losses = trainer.train(&tensor, self.training_config.epochs)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(losses)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Trainer not initialized"))
        }
    }

    fn predict<'py>(&self, py: Python<'py>, data: PyReadonlyArray2<f32>) -> PyResult<&'py PyArray1<u32>> {
        let array = data.as_array();
        let shape = array.shape();
        let n_samples = shape[0];
        let n_features = shape[1];

        // Convert to Candle tensor
        let flat_data: Vec<f32> = array.iter().cloned().collect();
        let tensor = Tensor::from_vec(flat_data, (n_samples, n_features), &self.device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        if let Some(trainer) = &self.trainer {
            let labels = trainer.model().predict(&tensor)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
            let labels_vec: Vec<u32> = labels.to_vec1()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
            Ok(PyArray1::from_vec(py, labels_vec))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not trained"))
        }
    }

    fn predict_proba<'py>(&self, py: Python<'py>, data: PyReadonlyArray2<f32>) -> PyResult<&'py PyArray2<f32>> {
        let array = data.as_array();
        let shape = array.shape();
        let n_samples = shape[0];
        let n_features = shape[1];

        // Convert to Candle tensor
        let flat_data: Vec<f32> = array.iter().cloned().collect();
        let tensor = Tensor::from_vec(flat_data, (n_samples, n_features), &self.device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        if let Some(trainer) = &self.trainer {
            let probs = trainer.model().predict_proba(&tensor)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
            let probs_vec: Vec<Vec<f32>> = probs.to_vec2()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
            // Flatten for PyArray2
            let flat_probs: Vec<f32> = probs_vec.into_iter().flatten().collect();
            let py_array = PyArray2::from_vec(py, flat_probs).reshape((n_samples, self.config.n_clusters))?;
            
            Ok(py_array)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not trained"))
        }
    }
}
