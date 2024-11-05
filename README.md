CODE GLOSSARY
1. MODELS AND ARCHITECTURES
GAUSSIAN PROCESS (GP) MODELS
gp_model: Primary GP model for TCNATT data analysis
Input: 12 features + reconstruction error
Output: Uncertainty estimates
gp_model_stacked: Secondary GP model for Stacked AE data
Input: 18 features + reconstruction error
Output: Uncertainty estimates for stacked architecture
NEURAL NETWORK MODELS
ae_model: Stacked Autoencoder for initial anomaly detection
Architecture: Multiple encoding/decoding layers
Input: 18-dimensional feature vectors
model (TCNATT): Temporal Convolutional Network with Attention
Features: 12-dimensional input
Temporal processing: Sequence-based analysis
2. BUFFER MANAGEMENT
MAIN BUFFERS
buffers: {Dict} Stores scenario-specific data sequences
Key format: "buffer_{i}" where i is scenario index
Value: List of feature vectors
buffer_stacked: List storing Stacked AE input data
Dimension: [n_samples, 18_features]
TIMESTAMP TRACKING
buffer_timestamps: {Dict} Timestamps for main buffers
Synchronized with buffers dictionary
buffer_stacked_timestamps: List of stacked buffer timestamps
Nanosecond precision timestamps
SEQUENCE PARAMETERS
sequence_length: Base sequence length for processing
gp_sequence_length: GP-specific sequence length (32)
stacked_sequence_length: Stacked AE sequence length
stride: Step size between consecutive sequences
step_size: Granularity of sequence creation
3. ANOMALY DETECTION AND TRACKING
STORAGE STRUCTURES
anomalous_sequences: {Dict} Detected anomalous patterns
Format: {buffer_name: [sequence_data]}
anomalous_errors: {Dict} Reconstruction errors
Format: {buffer_name: [error_values]}
anomalous_timestamps: {Dict} Anomaly occurrence times
Format: {buffer_name: [timestamp_values]}
STACKED SPECIFIC
stacked_anomalous_sequences: List of stacked AE anomalies
stacked_anomalous_errors: Associated error values
stacked_anomalous_timestamps: Occurrence timestamps
4. DATA PROCESSING AND MEASUREMENTS
FEATURE VECTORS
tcnatt_data: [Array] 12-dimensional feature vector
Components: [x1, y1, z1, x2, y2, z2, ...]
autoencoder_data: [Array] 18-dimensional feature vector
Extended feature set for stacked architecture
MEASUREMENT TRACKING
measurement_number: Current measurement ID
last_measurement_number: Previous measurement ID
server_timestamp: OPC UA server timestamp
Format: Nanoseconds since epoch
5. CONFIGURATION AND PARAMETERS
THRESHOLDS AND SETTINGS
thresholds: [Array] Anomaly detection thresholds
Values: [92, 95, 96, 98]
target_sizes: {Dict} Buffer size targets
Calculated based on scenarios
buffer_fill_percentage: Initial fill requirement
SCENARIOS
scenarios: [List] Parameter configurations
Format: [{"batch_size": x, "sequence_length": y}, ...]
6. STATE MANAGEMENT
INITIALIZATION FLAGS
initial_buffers_filled: {Dict} Buffer initialization status
stacked_initial_buffer_filled: Stacked buffer status
stacked_csv_loaded: CSV data loading status
OPERATIONAL COUNTERS
batch_counter: Processing batch counter
count_temp: Temperature warning counter
opc_reconnection_attempts: Connection retry counter
7. CONNECTIONS AND APIs
EXTERNAL CONNECTIONS
opc_client: OPC UA client connection
URL: opc.tcp://adslink228003766:16664/
write_api: InfluxDB write API
Database: test3
Organization: LL
8. UNCERTAINTY QUANTIFICATION
METRICS
uncertainties: [Array] GP uncertainty values
traffic_lights: [Array] Risk indicators
Values: ['red', 'yellow', 'green']
kl_divergence: KL divergence measure
elbo: Evidence Lower BOund
log_likelihood: Model log likelihood
9. SENSOR DATA
TEMPERATURE
temp: Current temperature reading
temp_old: Previous valid temperature
Fallback value for invalid readings
SPEED
speed_value: Raw speed measurement
speed_valueint: Integer speed value
POSITIONING
axis_info: Current axis identifier
axis_info_binary: Binary axis encoding
current_axis: Human-readable axis name
Values: ['x-axis', 'y-axis', 'z-axis']
10. FILE MANAGEMENT
PATHS AND STORAGE
csv_storage: CSV file directory path
configuration: {Dict} System configuration
Loaded from config.ini
11. KEY FUNCTIONS AND METHODS
INITIALIZATION
initialize_buffers(): Buffer setup with historical data
initialize_target_sizes(): Buffer size calculation
load_gp_model(): Model loading and initialization
DATA PROCESSING
fetch_data(): Main data collection loop
create_sequences(): Sequence generation
process_gp_data(): GP model processing
process_stacked_gp_data(): Stacked model processing
ANALYSIS
analyze_uncertainties(): Uncertainty analysis
assign_traffic_lights(): Risk level assignment
quantify_uncertainty_gp(): Uncertainty quantification
DATA MANAGEMENT
load_buffer_from_csv(): Historical data loading
save_buffer_to_csv(): Buffer data persistence
write_to_influx(): InfluxDB data writing
UTILITY
get_server_time(): Server time retrieval
cleanup(): Resource cleanup
push_gp_data_to_influx(): GP results storage
push_stacked_gp_to_influx(): Stacked GP results storage
12. ERROR HANDLING
TEMPERATURE SAFETY
Temperature threshold: 45°C
Maximum attempts: 3
Action: System shutdown on breach
OPC UA RECONNECTION
Maximum attempts: 100
Retry interval: 5 seconds
Action: System exit on max attempts
