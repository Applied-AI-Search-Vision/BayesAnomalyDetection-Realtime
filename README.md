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


### Sequence Parameters
- `sequence_length`: Base sequence length for processing
- `gp_sequence_length`: GP-specific sequence length (32)
- `stacked_sequence_length`: Stacked AE sequence length
- `stride`: Step size between consecutive sequences
- `step_size`: Granularity of sequence creation


### Stacked Architecture Specific
- `stacked_anomalous_sequences`: List of stacked AE anomalies
- `stacked_anomalous_errors`: Associated error values
- `stacked_anomalous_timestamps`: Occurrence timestamps

## Configuration and Parameters

### Thresholds and Settings
- **Thresholds**: `[92, 95, 96, 98]`
- **Buffer Fill Percentage**: Initial fill requirement
- **Target Sizes**: Calculated based on scenarios

### Uncertainty Quantification Metrics
- `uncertainties`: GP uncertainty values
- `traffic_lights`: Risk indicators ['red', 'yellow', 'green']
- `kl_divergence`: KL divergence measure
- `elbo`: Evidence Lower BOund
- `log_likelihood`: Model log likelihood

## Error Handling

### Temperature Safety
- **Threshold**: 45°C
- **Maximum attempts**: 3
- **Action**: System shutdown on breach

### OPC UA Reconnection
- **Maximum attempts**: 100
- **Retry interval**: 5 seconds
- **Action**: System exit on max attempts


## External Connections

### OPC UA
- **Client URL**: `opc.tcp://adslink228003766:16664/`

### InfluxDB
- **Database**: test3
- **Organization**: LL

## License

[MIT License](LICENSE)

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.
