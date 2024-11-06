# Real-time Anomaly Detection System

A real-time anomaly detection system using TCNATT and Stacked Autoencoder models with GP-based uncertainty quantification.

## Models and Architectures

### Gaussian Process (GP) Models

- **gp_model**: Primary GP model for TCNATT data analysis
  - Input: 12 features + reconstruction error
  - Output: Uncertainty estimates

- **gp_model_stacked**: Secondary GP model for Stacked AE data
  - Input: 18 features + reconstruction error
  - Output: Uncertainty estimates for stacked architecture

### Neural Network Models

- **ae_model**: Stacked Autoencoder for initial anomaly detection
  - Architecture: Multiple encoding/decoding layers
  - Input: 18-dimensional feature vectors

- **model (TCNATT)**: Temporal Convolutional Network with Attention
  - Features: 12-dimensional input
  - Temporal processing: Sequence-based analysis

## System Components

### Buffer Management

#### Main Buffers 
