from opcua import Client, ua
import time
import numpy as np
import torch
from DeploymentTCNATT import init_model, run_inference
import influxdb_client
from influxdb_client import Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import atexit
from pythonconfigtest import read_configuration
import serial
#import serial.tools.list_ports
import sys
import datetime
import traceback
import tensorflow as tf
import joblib
import os
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
import pandas as pd
from glob import glob
import serial



#First function of implementation loads the pretrained GP model for TCNATT & StackedAE. Future training should firsthand go to replace the current pth file here. 
def load_gp_model(save_path, save_path_stacked):
    checkpoint = torch.load(save_path, map_location=torch.device('cpu'))
    checkpoint_stacked = torch.load(configuration['StackedAE-GP_model_path'], map_location=torch.device('cpu')) #Loads the pth file found in the configuration script 
    inducing_points = checkpoint['inducing_points'].double() #Inducing points are points representative of the data by finding a subset of the data that is representative of the entire dataset. 
    inducing_points_stacked = checkpoint_stacked['inducing_points'].double()
   
    

   
    #First class definition. Source: https://arxiv.org/abs/1511.06499 
    class VariationalGPModel(ApproximateGP): #Due to inducing points we use an approximate GP model. Benefit is lower computational cost and faster inference. 
        def __init__(self, inducing_points):
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0)) #Cholesky is used to avoid jittering. Jitter = change of values in the covariance matrix. 
            variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
            super(VariationalGPModel, self).__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(1))) #IMPORTANT METRIC: The kernel we use. RBFKernel class is currently called on. Choice of kernel greatly affects model performance. 
            

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    gp_model = VariationalGPModel(inducing_points)
    gp_model.load_state_dict(checkpoint['gp_model_state_dict'])
    gp_model = gp_model.double() 

    gp_model_stacked = VariationalGPModel(inducing_points_stacked)
    gp_model_stacked.load_state_dict(checkpoint_stacked['gp_model_state_dict'])
    gp_model_stacked = gp_model_stacked.double()
    gp_model_stacked.eval()


    gp_model.eval()
    
    print(f"GP model dtype: {next(gp_model.parameters()).dtype}")
    return gp_model, gp_model_stacked



# KL Divergence calculation is used for approximating the difference in the prior gaussian distribution q from the posterior gaussian distribution p  
def kl_divergence_mvn(p, q):
     # p and q are multivariate normal distributions
     # p is typically the true/prior distribution
    # q is typically the approximating/variational distribution

    # Calculate the difference between means of distributions q and p
    mean_diff = q.mean - p.mean

    # Extract covariance matrices from both distributions
    cov_p, cov_q = p.covariance_matrix, q.covariance_matrix
    # Get the dimensionality of the distributions
    k = p.event_shape[0]  # Dimensionality
    
    # Create a small perturbation matrix (jitter) to ensure numerical stability
    jitter = 1e-6 * torch.eye(k, device=cov_p.device) #This creates a k×k identity matrix
    cov_p += jitter
    cov_q += jitter # Add the jitter to both covariance matrices to prevent numerical issues

    term1 = torch.logdet(cov_q) - torch.logdet(cov_p) # Term 1: log determinant difference
    term2 = torch.trace(torch.linalg.inv(cov_q) @ cov_p) # Term 2: trace term - measures the ratio of covariances
    term3 = mean_diff @ torch.linalg.inv(cov_q) @ mean_diff # Term 3: quadratic term - measures the distance between means
    

    return 0.5 * (term1 - k + term2 + term3) #This return statement is according to how the terms should be combined(KL divergence formula for multivariate normal distributions):
    # KL(p||q) = 0.5 * (log|Σq|/|Σp| - k + tr(Σq^(-1)Σp) + (μq-μp)^T Σq^(-1) (μq-μp)). SOURCE: https://statproofbook.github.io/P/mvn-kl.html

# Uncertainty quantification using GP
def quantify_uncertainty_gp(gp_model, normalized_anomalies):
    gp_model.eval()
    uncertainties = [] 
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var(): #Used to calibrate the calculation of the posterior distribution. SOURCE: https://arxiv.org/abs/1803.06058
        for sequence in normalized_anomalies:
            # Reshape the sequence to (1, n) where n is the sequence length
            x = sequence.view(1, -1)  # view(1, -1) makes it a 2D tensor with 1 row and automatically calculated columns (that's why -1)
            try:
                prior_dist = gp_model.prior(x) # Get the prior distribution from the GP model
                posterior_dist = gp_model(x) # Get the posterior distribution by running the sequence through the GP model
                kl_div = kl_divergence_mvn(posterior_dist, prior_dist) # Calculate KL divergence between posterior and prior distributions
                # Higher KL divergence indicates higher uncertainty
                uncertainties.append(kl_div.item()) #Here we convert the KL-Divergence ourput into a python scalar value we can interpret for analysis 
            except Exception as e:
                print(f"Error in GP_Inference: {e}")
                uncertainties.append(float('nan'))

    
    return uncertainties


#This function is used to send the log files of terminal output to an email
def send_log_email(log_filename, recipient_email):
    sender_email = "stellanlange@gmail.com" 
    password = "avsxkpolzqxlzjec"
    
    
    logs_dir = 'logs'
    log_path = os.path.join(logs_dir, log_filename)
    
    
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = f"Log File: {log_filename}"
    
    body = "Please find attached the latest log file."
    message.attach(MIMEText(body, "plain"))
    
    try:
        with open(log_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {log_filename}",
        )
        message.attach(part)
        
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.send_message(message)
        #print(f"Log file sent to {recipient_email}") #Uncomment to get a confirmation in the terminal output every time an email is sent
    except Exception as e:
        print(f"Error sending email: {e}")



current_time = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S") #current time 
log_filename = f"log_{current_time}.txt" #specifies what the filename of the log file should be 

#The teelogger class is used to send standard output and error output to the log file we save and to the terminal at the same time 
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        
        
        logs_dir = 'logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        
        log_path = os.path.join(logs_dir, filename)
        self.log = open(log_path, "w")
        
        self.last_message_had_newline = True

    def write(self, message):
        lines = message.splitlines(True)
        for i, line in enumerate(lines):
            if self.last_message_had_newline or i > 0:
                if not line.startswith('[20'): 
                    current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    line = f"[{current_time} UTC] {line}"
            self.terminal.write(line)
            self.log.write(line)
        
        self.last_message_had_newline = message.endswith('\n')
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = TeeLogger(log_filename)
sys.stderr = TeeLogger(log_filename) 

#count_temp = 0

# Arduino setup
#arduino_port = ""
#ports = serial.tools.list_ports.comports()
#for port, desc, hwid in sorted(ports):
    #if "Arduino" in desc:
        #print("Found Arduino on port = ", port)
        #arduino_port = port

#try:
    #dev = serial.Serial(arduino_port, baudrate=9600, timeout=1.5)
#except:
    #print("Failed to establish connection with Arduino")

config_file_path = 'configtest.ini'
configuration = read_configuration(config_file_path)

opc_client_url = configuration['opc_client_url']
influxdb_url = configuration['influxdb_url']

last_measurement_number = None
batch_counter = 0
opc_reconnection_attempts = 0

def windowing(data, window_size):
    # Step 2: Sliding window for the Autoencoder 
    return np.lib.stride_tricks.sliding_window_view(data, (window_size, data.shape[1])).reshape(-1, window_size * data.shape[1])

def stacked_ae_inference(data, ae_model, window_size, kernel='rbf', nu=0.15, gamma='scale'): #Nu value is the most important parameter for model performance, lowering the nu value will make the model less sensitive to categorizing a datapoint as an outlier 
    try:
        # Step 1: Windowing the data
        windowed_data = windowing(data, window_size)
        
        # Step 2: Autoencoder Reconstruction
        input_tensor = tf.constant(windowed_data, dtype=tf.float32)
        reconstructions = ae_model(input_tensor)


        #Below we are checking  so that we properly access pretrained stacked autoencoder model output 
        if isinstance(reconstructions, dict):
            reconstructions = reconstructions['output_1'] # we extract the output from the dictionary, the dictionary output is the output of the autoencoder. 
        
        # Convert reconstructions to NumPy array
        reconstructions = reconstructions.numpy()
        
        # Step 3: Calculating Residuals
        if windowed_data.shape != reconstructions.shape:
            print(f"Shape mismatch: windowed_data {windowed_data.shape}, reconstructions {reconstructions.shape}")
            raise ValueError("Shape mismatch between input and reconstruction")
        
        #EXPLAIN:residual_val = ae_predict(val_data, model_parameters['stacked']) - val_data is the original implementation of residual we wish to use in realtime
        #we subtract the reconstruction(model output) from the raw input data to get the residual
        residuals = reconstructions - windowed_data #Residuals are compared to the original input, reconstructions by itself does not make that comparison, so 0.5 in value would be without context. 
        
        # Step 4: Fitting the OCSVM
        reshaped_residuals = residuals.reshape(residuals.shape[0], -1)
        svm_model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        svm_model.fit(reshaped_residuals)
        
        # Step 5: Anomaly Detection
        #svm_decision = svm_model.decision_function(reshaped_residuals) # the distance to the hyperplane, i.e checking distance of inliers vs outlier datapoints
        pred_labels = svm_model.predict(reshaped_residuals)
        
        # Transform labels: 1 for anomaly, 0 for normal
        pred_labels = np.where(pred_labels == 1, 0, 1)
        
        return reconstructions, residuals, pred_labels, windowed_data
    except Exception as e:
        print(f"[ERROR] Failed during stacked_ae_inference: {e}")
        traceback.print_exc()
        return None, None, None, None, None


#Our CodeLogic class contains all initializations of lists, dictionaries, variables, other forms of storage objects and important functions and code logic in its sub-function called Fetch_data 
class CodeLogic:
    def __init__(self, write_api, opc_client, gp_model, gp_model_stacked, thresholds):
        self.write_api = write_api #used for influxDB
        self.opc_client = opc_client #used for the OPC server where dta ais fetched from
        self.gp_model = gp_model #initiate the gp model as a class object 
        self.ae_model = tf.keras.layers.TFSMLayer('C:\\Users\\lladmin\\Desktop\\Deployment\\autoencoder_model\\ae_model_0', call_endpoint='serving_default')
        self.scaler = joblib.load('C:\\Users\\lladmin\\Desktop\\Deployment\\autoencoder_model\\scaler.joblib')
        self.window_size = 16 #window size is used for the stacked autoencoder 
        self.buffer_stacked = [] #buffer for ae raw feature data 
        self.buffer_stacked_timestamps = []
        self.gp_model_stacked = gp_model_stacked
        self.stacked_anomalous_sequences = [] #used for AE-GP to store raw feature data 
        self.stacked_anomalous_errors = [] #residuals are appended here
        self.stacked_anomalous_timestamps = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = init_model() # Initialize the TCN-ATT model
        self.model.to(self.device)
        self.stacked_csv_loaded = False
        self.stacked_loaded_data = None
        self.scenarios = scenarios
        self.initial_buffers_filled = [False] * len(scenarios)
        self.csv_storage = 'Buffer Data' #We create a folder where we store CSV files of data 
        if not os.path.exists(self.csv_storage):
            os.makedirs(self.csv_storage)
        
        self.tcnatt_data = [-1] * 12
        self.autoencoder_data = [-1] * 18
        self.buffer_stacked = []
        self.buffer_stacked_timestamps = []
        self.stacked_sequence_length = 288
        self.buffer_fill_percentage = 0.99999 #This variable controls how much of CSV files data we should read into the buffers (percentage)
        self.stacked_gp_sequence_length = configuration["gp_sequence_length"] #current amount of anomlaies for the GP
        self.step_size = 1 
        self.buffers_initialised = False
        self.initial_buffer_filled = False
        self.gp_initial_buffer_filled = False
        self.stacked_initial_buffer_filled = False
        self.thresholds = thresholds
        self.buffers = {f"buffer_{i}": [] for i in range(len(scenarios))} #This is the primary buffer for TCNATT, one buffer for each scenario 
        self.buffer_timestamps = {f"buffer_{i}": [] for i in range(len(scenarios))}
        self.gp_buffers = {f"gp_buffer_{i}": [] for i in range(len(scenarios))} #Buffer for the GP model so that GP_buffer_0 corresponds to scenario 0 for the TCNATT model 
        self.gp_buffer_timestamps = {f"gp_buffer_{i}": [] for i in range(len(scenarios))}
        
        
        # GP-specific initializations
        self.gp_sequence_length = configuration["gp_sequence_length"]
        self.anomalous_sequences = {}
        self.anomalous_errors = {}
        self.anomalous_timestamps = {}
        
        
        for i in range(len(scenarios)):
            self.anomalous_sequences[f"buffer_{i}"] = [] #This list stores the feature data used as nput for the GP model
            self.anomalous_errors[f"buffer_{i}"] = [] #This list stores the error reconstruction values used as input fort the GP model
            self.anomalous_timestamps[f"buffer_{i}"] = []
        
        self.initialize_target_sizes()
        self.initialize_buffers()
    
   
    
    def initialize_target_sizes(self):
        """
        Initializes target sizes for each scenario's data buffer based on:
        1. Scenario parameters (batch_size and sequence_length)
        2. Historical data from CSV files
        3. Buffer fill percentage requirement
        """
        self.target_sizes = {}
        for i, scenario in enumerate(self.scenarios):
            batch_size = scenario["batch_size"]
            sequence_length = scenario["sequence_length"]
            stride = batch_size * sequence_length
            csv_data = self.load_buffer_from_csv(i) 
            self.target_sizes[i] = min(int(stride * self.buffer_fill_percentage), len(csv_data)) #Get the available historical data using min 
    
    #initialize buffers with the calculated target sizes. This function is responsible to distribute the correct amount of data to each buffer depending on the size of the buffer
    def initialize_buffers(self):
        """Initialize all buffers with historical data from CSV files"""
        print("Initializing buffers with historical data...")
        
        # Initialize TCNATT buffers
        for i, _ in enumerate(scenarios):
            buffer_name = f"buffer_{i}"
            target_size = self.target_sizes[i]
            loaded_data = self.load_buffer_from_csv(i)
            if loaded_data:
                data_to_load = loaded_data[:min(target_size, len(loaded_data))]
                if len(data_to_load) != target_size:
                    print(f"Warning: Loaded {len(data_to_load)} entries instead of target {target_size} for buffer {i}")
                
                # Initialize main buffer
                self.buffers[buffer_name] = data_to_load.copy()
                self.buffer_timestamps[buffer_name] = [time.time_ns() for _ in data_to_load]
                print(f"Buffer {i} initialized with {len(data_to_load)} entries from CSV")
            else:
                print(f"No historical data found for buffer {i}")

        # Initialize Stacked AE buffer if needed
        stacked_data = self.load_buffer_from_csv("stacked_ae")
        if stacked_data:
            target_size = int(self.stacked_sequence_length * self.buffer_fill_percentage)
            data_to_load = stacked_data[:target_size]
                
            self.buffer_stacked = data_to_load.copy()
            self.buffer_stacked_timestamps = [time.time_ns() for _ in data_to_load]
            self.stacked_csv_loaded = True
            print(f"Stacked AE buffer initialized with {len(data_to_load)} entries from CSV")
        else:
            print("No historical data found for stacked AE buffer")


    #Fetch data is our data pipeline orchestration function. 
    def fetch_data(self):
        global last_measurement_number, batch_counter, temp_old, count_temp, opc_reconnection_attempts
        try:
            # Fetch data from OPC UA server
            try:
                node = self.opc_client.get_node("ns=1;s=0xA0") #This is the node we use to fetch data from the OPC server 
                values = node.get_value()
                opc_reconnection_attempts = 0
            except:
                print("Failed to establish connection to OPC_UA SERVER")
                print("trying to recconect")
                time.sleep(5)
                #dev.write(b'R\n')
                opc_client.connect()
                opc_reconnection_attempts = opc_reconnection_attempts + 1
                if opc_reconnection_attempts > 100:
                    print("Max attempts, shutting of!")
                    sys.exit()
                return

            #Extract relevant values from the binary string
            int_value = values[0]
            binary_value = format(int_value, '032b')
            measurement_number = int(binary_value[-16:], 2)

            if measurement_number == last_measurement_number:
                #dev.write(b'R\n')
                return
            
            # Get current timestamp
            server_timestamp = get_server_time(self.opc_client)

            speed_value = values[3] / 65536 * 60
            speed_valueint = int(speed_value)

            #Code pertaining to temperature is specific to an Arduino device registering temperature
            #if 'temp_old' not in globals():
                #global temp_old
                #temp_old = None

            #dev.reset_input_buffer()
            #temp = dev.readline()
            #if "S" not in str(temp):
                #temp = temp_old if temp_old is not None else "0.0"
            #else:
                #temp = str(temp).split("S",1)[1]
                #temp = temp.split("\\",1)[0]
                #try:
                    #if float(temp) == -127.00:
                        #temp = temp_old if temp_old is not None else "0.0"
                    #else:
                        #var = opc_client.get_node("ns=1;s=%s" % '0xAF')
                        #var.set_value(ua.Variant(float(temp), ua.VariantType.Float))
                        #temp_old = temp
                #except ValueError:
                    #temp = temp_old if temp_old is not None else "0.0"

            #if int(float(temp)) > 45:
                #count_temp += 1
                #print("Temperature too high!")
                #print(f'Temp = {temp}')
                #if count_temp > 3:
                    #print("Temperature too high, shutting off!")
                    #print(f'Temp! = {temp}')
                    #dev.write(b'S\n')
                    #print(time.strftime("%Y-%m-%d\n%H:%M:%S\n", time.localtime()))
                    #sys.exit()
            #else:
                #count_temp = 0

            axis_info_binary = binary_value[-19:-16]
            axis_info = int(axis_info_binary, 2)
            axis_map = {1: "x-axis", 2: "y-axis", 4: "z-axis"}
            current_axis = axis_map.get(axis_info, "unknown")
            
            print(f"Measurement number: {measurement_number}, Current Axis: {current_axis}")
          
            last_measurement_number = measurement_number

            relevant_values = values[8:26] 
            converted_values = [val / 65536 for val in relevant_values]

            if axis_info == 1:
                self.tcnatt_data[0] = converted_values[0]
                self.autoencoder_data[0] = converted_values[0]
                self.tcnatt_data[3] = converted_values[1]
                self.autoencoder_data[3] = converted_values[1]
                self.tcnatt_data[6] = converted_values[2]
                self.autoencoder_data[6] = converted_values[2]
                self.tcnatt_data[9] = converted_values[3]
                self.autoencoder_data[9] = converted_values[3]
                self.autoencoder_data[12] = converted_values[4]
                self.autoencoder_data[15] = converted_values[5]
            elif axis_info == 2:
                self.tcnatt_data[1] = converted_values[0]
                self.autoencoder_data[1] = converted_values[0]
                self.tcnatt_data[4] = converted_values[1]
                self.autoencoder_data[4] = converted_values[1]
                self.tcnatt_data[7] = converted_values[2]
                self.autoencoder_data[7] = converted_values[2]
                self.tcnatt_data[10] = converted_values[3]
                self.autoencoder_data[10] = converted_values[3]
                self.autoencoder_data[13] = converted_values[4]
                self.autoencoder_data[16] = converted_values[5]
            else:
                self.tcnatt_data[2] = converted_values[0]
                self.autoencoder_data[2] = converted_values[0]
                self.tcnatt_data[5] = converted_values[1]
                self.autoencoder_data[5] = converted_values[1]
                self.tcnatt_data[8] = converted_values[2]
                self.autoencoder_data[8] = converted_values[2]
                self.tcnatt_data[11] = converted_values[3]
                self.autoencoder_data[11] = converted_values[3]
                self.autoencoder_data[14] = converted_values[4]
                self.autoencoder_data[17] = converted_values[5]

            if -1 not in self.tcnatt_data and -1 not in self.autoencoder_data:
                self.buffer_stacked.append(self.autoencoder_data)
                self.buffer_stacked_timestamps.append(server_timestamp)

                tcnatt_fields = {f"value_{i}": val for i, val in enumerate(self.tcnatt_data)}
                autoencoder_fields = {f"value_{i}": val for i, val in enumerate(self.autoencoder_data)}
                #We call upon the influx function handling the write operation for the raw feature data 
                write_to_influx(self.write_api, "Raw_Data", tcnatt_fields, tags={"model_type": "TCNATT"}, timestamp=server_timestamp)
                write_to_influx(self.write_api, "Raw_Data", autoencoder_fields, tags={"model_type": "Autoencoder"}, timestamp=server_timestamp)

                speed_fields = {"speed": speed_valueint}
                write_to_influx(self.write_api, "Speed_data", speed_fields, timestamp=server_timestamp)

                # Write temperature data
                #temp_fields = {"temperature": float(temp)}
                #temp_timestamp = int(time.time() * 1e9)
                #write_to_influx(self.write_api, "Temperature_data", temp_fields, timestamp=temp_timestamp)

                #We append data from the OPC server to the model specific buffers 
                for i, scenario in enumerate(scenarios):
                    buffer_name = f"buffer_{i}"
                    self.buffers[buffer_name].append(self.tcnatt_data.copy())
                    self.buffer_timestamps[buffer_name].append(server_timestamp)
                    scenario = scenarios[i]
                    batch_size = scenario["batch_size"]
                    sequence_length = scenario["sequence_length"]
                    stride = batch_size * sequence_length
                    buffer_size = len(self.buffers[buffer_name])

                    target_size = self.target_sizes[i]
                
                
                    
                    if not self.initial_buffers_filled[i]: #Here a flag-based system (boolean logic) starts. When this is True we exit the initial data processing logic 
                        # Initial buffer filling phase begins...
                       
                            
                        buffer_size = len(self.buffers[buffer_name])
                        percentage = int((buffer_size / stride) * 100)
                        print(f'TCNATT-Buffer: {i} filled with {buffer_size} out of {stride} ({percentage}%)')

                        if buffer_size == stride: #stride is calculated as batch size multiplied by sequence length 
                            print(f"Initial buffer is now filled for buffer: {i}")
                                # Save buffer to CSV if needed - uncomment if needed 
                                #self.save_buffer_to_csv(self.buffers[buffer_name], i)
                                # Create sequences for model inference
                            self.initial_buffers_filled[i] = True #Enough data is collected for the initial data processing block 
                            batch_data = create_sequences(self.buffers[buffer_name], sequence_length, self.step_size)
                            batch_tensor = torch.tensor(batch_data)
                            batch_tensor = batch_tensor.permute(0, 2, 1)
                                
                            try:
                                output = run_inference(self.model, batch_tensor)
                                output = output.transpose(1, 2)
                                reconstruction_error = ((batch_tensor - output) ** 2).mean(dim=(1, 2)).cpu().numpy()

                                

                                parameter_settings = f"batch_size={batch_size}_seq_length={sequence_length}"
                                # Loop over reconstruction error to push to InfluxDB
                                for j in range(len(reconstruction_error)):
                                    anomaly_timestamp = self.buffer_timestamps[buffer_name][j]

                                    data_point = Point("TCNATTPredictions")
                                    data_point.tag("parameter_settings", parameter_settings)
                                    data_point.time(anomaly_timestamp, WritePrecision.NS)
                                    data_point.field("reconstruction_error", float(reconstruction_error[j]))

                            

                                    for threshold in self.thresholds:
                                        threshold_value = np.percentile(reconstruction_error, threshold)
                                        anomaly_detected = int(reconstruction_error[j] > threshold_value)
                                        data_point.field(f"anomaly_detected_{threshold}", anomaly_detected)
                                        self.write_api.write(bucket=bucket, org=org, record=data_point)
                                            
                         
                                            
                                        

                                    # Check if the datapoint is an anomaly
                                    if int(reconstruction_error[j] > np.percentile(reconstruction_error, 92)):
                                        self.anomalous_sequences[buffer_name].append(self.buffers[buffer_name][j]) #append raw data 
                                        self.anomalous_errors[buffer_name].append(reconstruction_error[j]) #append reconstruction errors
                                        self.anomalous_timestamps[buffer_name].append(anomaly_timestamp)
           
                                        if len(self.anomalous_sequences[buffer_name]) > self.gp_sequence_length: #condition to check if we have amassed more data then our volume size for GP anomalies (specified in the beginning of the main class)
                                            self.anomalous_sequences[buffer_name] = self.anomalous_sequences[buffer_name][1:] #If condition is true we have too much data for our input and remove the oldest value - The start of the sliding window effect for realtme processing for the TCNATT-GP model
                                            self.anomalous_errors[buffer_name] = self.anomalous_errors[buffer_name][1:]
                                            self.anomalous_timestamps[buffer_name] = self.anomalous_timestamps[buffer_name][1:]


                                        gp_buffer_size = len(self.anomalous_sequences[buffer_name])
                                        percentage = int((gp_buffer_size / self.gp_sequence_length) * 100)
                                        print(f'GP-Buffer_{i} filled with {gp_buffer_size} out of {self.gp_sequence_length} ({percentage}%)')
                                    
                                if len(self.anomalous_sequences[buffer_name]) == self.gp_sequence_length:
                                    print(f"=== GP Buffer {i} Initialized and Ready for Realtime Processing ===")
                                    gp_data = np.array(self.anomalous_sequences[buffer_name])
                                    gp_errors = np.array(self.anomalous_errors[buffer_name])
                                    gp_timestamps = self.anomalous_timestamps[buffer_name]

                                            
                                    uncertainties, thresholds_gp, traffic_lights = self.process_gp_data(gp_data, gp_errors)

                                    # Add traffic light distribution summary
                                    light_counts = {
                                            "green": traffic_lights.count("green"),
                                            "yellow": traffic_lights.count("yellow"),
                                            "red": traffic_lights.count("red")
                                        }
                                    print("\nTraffic Light Distribution TCNATT-GP:")
                                    print(f"  Green:  {light_counts['green']:2d} anomalies")
                                    print(f"  Yellow: {light_counts['yellow']:2d} anomalies")
                                    print(f"  Red:    {light_counts['red']:2d} anomalies")
                                    print(f"  Total:  {sum(light_counts.values()):2d} anomalies\n")
                                    for idx in range(len(uncertainties)): #We iterate through all uncertainty values and write to influx. 
                                        self.push_gp_data_to_influx(i, gp_timestamps[idx], uncertainties[idx], traffic_lights[idx], gp_errors[idx], thresholds_gp)
                                                        
                                    #We remove the oldest datapoint to maintain a sliding window effect of one datapoint
                                    self.anomalous_sequences[buffer_name] = self.anomalous_sequences[buffer_name][1:] 
                                    self.anomalous_errors[buffer_name] = self.anomalous_errors[buffer_name][1:]
                                    self.anomalous_timestamps[buffer_name] = self.anomalous_timestamps[buffer_name][1:]
                                    
                                        
                                    
                                #print(f"=== Completed Initial Phase for Buffer {i} ===\n")
                                self.buffers[buffer_name]=self.buffers[buffer_name][1:]
                                self.buffer_timestamps[buffer_name]=self.buffer_timestamps[buffer_name][1:]
 

                            except Exception as inference_error:
                                print(f"An error occurred during TCN-ATT inference: {inference_error}")
                                traceback.print_exc()


                    ################################################# REALTIME PROCESSING ############################################################

                    else:
                        # Real-time processing mode starts for the TCNATT-GP

                        buffer_size = len(self.buffers[buffer_name])
                        percentage = int((buffer_size / stride) * 100)

                        if buffer_size == stride:
                            print(f'TCNATT-Buffer:{i} full in real-time processing mode with {buffer_size} out of {stride} ({percentage}%)')

                            # Create sequences for model inference
                            batch_data = create_sequences(self.buffers[buffer_name], sequence_length, self.step_size)
                            

                            batch_tensor = torch.tensor(batch_data)
                            batch_tensor = batch_tensor.permute(0, 2, 1)

                            try:
                                #print("\n=== TCNATT Realtime Inference Initiated ===")
                                output = run_inference(self.model, batch_tensor)
                                output = output.transpose(1, 2)
                                reconstruction_error = ((batch_tensor - output) ** 2).mean(dim=(1, 2)).cpu().numpy()

                                # Calculate the reconstruction error for the newest datapoint
                                reconstruction_error_shortened = reconstruction_error[-1]                                
                                newest_value = self.buffers[buffer_name][-1]
                                newest_timestamp = self.buffer_timestamps[buffer_name][-1]

                                # Prepare data for InfluxDB
                                anomaly_timestamp = newest_timestamp
                                parameter_settings = f"batch_size={batch_size}_seq_length={sequence_length}"

                                data_point = Point("TCNATTPredictions")
                                data_point.tag("parameter_settings", parameter_settings)
                                data_point.time(anomaly_timestamp, WritePrecision.NS)
                                data_point.field("reconstruction_error", float(reconstruction_error_shortened))
                                # Writing to InfluxDB
                                

                                # Determine anomaly based on threshold
                                for threshold in self.thresholds:
                                    threshold_value = np.percentile(reconstruction_error, threshold)
                                    anomaly_detected = int(reconstruction_error_shortened > threshold_value)
                                    data_point.field(f"anomaly_detected_{threshold}", anomaly_detected)
                                    
                                    if anomaly_detected == 1:
                                        print(f"Anomaly timestamp = {anomaly_timestamp}")
                                    
                                self.write_api.write(bucket=bucket, org=org, record=data_point)

                                if int(reconstruction_error_shortened > np.percentile(reconstruction_error, 92)):
                                    print(f"Found anomaly in realtime processing at time: {anomaly_timestamp}")
                                    self.anomalous_sequences[buffer_name].append(newest_value)
                                    self.anomalous_errors[buffer_name].append(reconstruction_error_shortened)
                                    self.anomalous_timestamps[buffer_name].append(anomaly_timestamp)
                                    
                                    # GP Model Processing
                                    gp_buffer_size = len(self.anomalous_sequences[buffer_name])
                                    percentage = int((gp_buffer_size / self.gp_sequence_length) * 100)
                                    print(f'GP-Buffer_{i} filled with {gp_buffer_size} out of {self.gp_sequence_length} ({percentage}%)')


                                    if gp_buffer_size > self.gp_sequence_length:
                                        print("MEGAERROR BUFFER TOO LONG")

                                    if gp_buffer_size == self.gp_sequence_length:
                                        print(f"=== Real-time Phase TCNATT-GP Processing ===")
                                        
                                        

                                        gp_data = np.array(self.anomalous_sequences[buffer_name])
                                        gp_errors = np.array(self.anomalous_errors[buffer_name])
                                        gp_timestamps = self.anomalous_timestamps[buffer_name]
                                      
                                        uncertainties, thresholds_gp, traffic_lights = self.process_gp_data(gp_data, gp_errors)


                                        # Push only the latest value to InfluxDB
                                        latest_idx = -1
                                        self.push_gp_data_to_influx(
                                            i, 
                                            gp_timestamps[latest_idx], 
                                            uncertainties[latest_idx], 
                                            traffic_lights[latest_idx], 
                                            gp_errors[latest_idx], 
                                            thresholds_gp
                                        )

                                        
                                        print(f"Uncertainty: {uncertainties[latest_idx]:.6f}")
                                        print(f"Traffic Light: {traffic_lights[latest_idx]}")

                                        # After processing, trim the buffer to maintain the most recent anomalies
                                        
                                        self.anomalous_sequences[buffer_name] = self.anomalous_sequences[buffer_name][1:]
                                        self.anomalous_errors[buffer_name] = self.anomalous_errors[buffer_name][1:]
                                        self.anomalous_timestamps[buffer_name] = self.anomalous_timestamps[buffer_name][1:]
                                      
                            # Remove the oldest datapoint from the buffer to maintain stride
                                self.buffers[buffer_name] = self.buffers[buffer_name][1:]
                                self.buffer_timestamps[buffer_name] = self.buffer_timestamps[buffer_name][1:]

                            except Exception as inference_error:
                                print(f"An error occurred during inference: {inference_error}")
                                traceback.print_exc()
                # End of scenarios loop
                self.tcnatt_data = [-1] * 12
                self.autoencoder_data = [-1] * 18
                    
                     ########################################## STACKED ENCODER ########################################################################
                if not self.stacked_initial_buffer_filled:

                    if len(self.buffer_stacked) == self.stacked_sequence_length:
                        print(f"AE Initial buffer fill complete with {len(self.buffer_stacked)} datapoints")

                        # Process all data points for the initial fill
                        buffer_array = np.array(self.buffer_stacked)
                        normalized_data = self.scaler.transform(buffer_array)
                        print('Processing initial stacked buffer')
                        print('Starting inference for Stacked AE')
                        self.stacked_initial_buffer_filled = True

                        # Run stacked autoencoder inference
                        reconstructions, residuals, pred_labels, windowed_data = stacked_ae_inference(
                            normalized_data,
                            self.ae_model,
                            self.window_size,
                        )

                        print('Inference done for stacked AE')
                        total_anomalies = sum(pred_labels==1)

                        print(f"Total number of anamalies =  {total_anomalies}")

                        #Each reconstruction is a 288 dimensional array, if we do not use residals here we take the mean value of each 18 features x 16 timesteps value.  
                    
                        # Write all initial anomalies to InfluxDB
                        num_windows = len(pred_labels) 
                        for i in range(num_windows):  
                            anomaly_label = int(pred_labels[i] == 1)  # 1 for anomaly, 0 for normal
                            reconstruction_error = float(np.mean(reconstructions[i]))
                            timestamp = self.buffer_stacked_timestamps[i]


                            point = Point("StackedAEPredictions") \
                                .tag("model", "stacked_autoencoder") \
                                .field("anomaly", anomaly_label) \
                                .field("reconstruction_error", reconstruction_error) \
                                .time(timestamp)
                            self.write_api.write(bucket=bucket, org=org, record=point)

                            if pred_labels[i] == 1: #Pred labels stores the predictions made by the SVM module, as data bein ginliers or outliers (0 for inlier)
                                self.stacked_anomalous_sequences.append(windowed_data[i])
                                self.stacked_anomalous_errors.append(residuals[i])
                                self.stacked_anomalous_timestamps.append(self.buffer_stacked_timestamps[i])

                            if len(self.stacked_anomalous_sequences)>self.stacked_gp_sequence_length:
                                self.stacked_anomalous_sequences=self.stacked_anomalous_sequences[1:]
                                self.stacked_anomalous_errors=self.stacked_anomalous_errors[1:]
                                self.stacked_anomalous_timestamps=self.stacked_anomalous_timestamps[1:]
                    
                        print(f"Initial GP buffer filled with {len(self.stacked_anomalous_sequences)} anomalies")

                        # Trim the buffer to maintain the desired size
                        
                        
                        self.buffer_stacked = self.buffer_stacked[1:]
                        self.buffer_stacked_timestamps = self.buffer_stacked_timestamps[1:]

                        #######Initial Stacked AE-GP condition criteria and processing starts here###########

                        if len(self.stacked_anomalous_sequences) == self.stacked_gp_sequence_length:
                            #print('Starting Stacked AE-GP Inference')
                            uncertainties, thresholds, traffic_lights = self.process_stacked_gp_data(
                                self.stacked_anomalous_sequences,
                                self.stacked_anomalous_errors,
                                self.stacked_anomalous_timestamps
                            )
                   
                            print("\nGP-Stacked Analysis Summary For Initial Run:")
                            print(f"Mean uncertainty: {thresholds['mean']:.6f}")
                            print(f"Standard deviation: {thresholds['mean_plus_std'] - thresholds['mean']:.6f}")
                            print(f"Range: [{np.min(uncertainties):.6f} - {np.max(uncertainties):.6f}]")
                            print(f"ELBO: {thresholds['elbo']:.6f}")
                            print(f"KL Divergence: {thresholds['kl_divergence']:.6f}")
                            
                            light_counts = {
                                "green": traffic_lights.count("green"),
                                "yellow": traffic_lights.count("yellow"),
                                "red": traffic_lights.count("red")
                            }
                            
                            print("\nTraffic Light Distribution GP-Stacked:")
                            for color, count in light_counts.items():
                                print(f"  {color.capitalize()}: {count}")
                            
                            
                            self.stacked_anomalous_sequences = self.stacked_anomalous_sequences[1:]
                            self.stacked_anomalous_errors = self.stacked_anomalous_errors[1:]
                            self.stacked_anomalous_timestamps = self.stacked_anomalous_timestamps[1:]
                            
                else:
                    print(f'AE-Buffer full in real-time processing mode with {len(self.buffer_stacked)} out of {self.stacked_sequence_length} ({float(len(self.buffer_stacked)*100/self.stacked_sequence_length)}%)')
                    # Real-time processing phase
                    
                    if len(self.buffer_stacked) == self.stacked_sequence_length:
                        # Process data and get predictions
                        buffer_array = np.array(self.buffer_stacked)
                        normalized_data = self.scaler.transform(buffer_array)
                        
                        # Run stacked autoencoder inference
                        reconstructions, residuals, pred_labels, windowed_data = stacked_ae_inference(
                            normalized_data,
                            self.ae_model,
                            self.window_size,
                        )
                        
                        latest_anomaly = np.sum(pred_labels[-1] == 1)
                        print(f"Latest Anomaly_status in real-time: {latest_anomaly} ")
                        latest_reconstruction_error = float(reconstructions[-1][0]) 
                        latest_timestamp = self.buffer_stacked_timestamps[-1]
                       
                        
                        # Process anomalies if found
                        
                        if latest_anomaly == 1:
                        
                            print("Processing detected anomaly for Stacked AE:")
                       
                            #The newest datapoint is appended and processed         
                            self.stacked_anomalous_sequences.append(windowed_data[-1]) 
                            self.stacked_anomalous_errors.append(residuals[-1])
                            self.stacked_anomalous_timestamps.append(self.buffer_stacked_timestamps[-1])
                            #if we have collected enough data to meet our volume criteria --> Inference 
                            if len(self.stacked_anomalous_sequences) == self.stacked_gp_sequence_length:
                                uncertainties, thresholds, traffic_lights = self.process_stacked_gp_data(self.stacked_anomalous_sequences,self.stacked_anomalous_errors,self.stacked_anomalous_timestamps)
                                print(f"Uncertainty: {uncertainties[-1]}")
                                print(f"Traffic Light: {traffic_lights[-1]}")
                                
                                #We write the newest datapoint to influx 
                                self.push_stacked_gp_to_influx(
                                    self.stacked_anomalous_timestamps[-1],
                                    uncertainties[-1],
                                    traffic_lights[-1],
                                    self.stacked_anomalous_errors[-1],
                                    thresholds) 

                                
                            #The oldest value is removed to maintain a sliding window approach 
                            self.stacked_anomalous_sequences=self.stacked_anomalous_sequences[1:]
                            self.stacked_anomalous_errors=self.stacked_anomalous_errors[1:]
                            self.stacked_anomalous_timestamps=self.stacked_anomalous_timestamps[1:]  

                        
                        #Data written to influx 
                        stacked_ae_point = Point("StackedAEPredictions") \
                                    .tag("model", "stacked_autoencoder") \
                                    .field("anomaly_detected", latest_anomaly) \
                                    .field("reconstruction_error", float(latest_reconstruction_error)) \
                                    .time(latest_timestamp)
                        self.write_api.write(bucket=bucket, org=org, record=stacked_ae_point)    
                      
                    
                    self.buffer_stacked=self.buffer_stacked[1:]
                    self.buffer_stacked_timestamps=self.buffer_stacked_timestamps[1:]
    
    

        except Exception as e:
            print(f"An error occurred in fetch_data: {e}")
            traceback.print_exc()
    #Buffer logic ends here and GP model code begins ending with cleanup code and finally statement 
                             
                             

    def process_gp_data(self, gp_data, gp_errors):
        """
        Process data through the Gaussian Process model to generate uncertainty estimates and traffic light indicators.
        
        Args:
            gp_data: Array of sequences containing anomalous patterns
            gp_errors: Array of reconstruction errors associated with each sequence
            
        Returns:
            tuple: (uncertainties, thresholds, traffic_lights)
                - uncertainties: Array of uncertainty values for each input
                - thresholds: Dictionary of threshold values for traffic light assignment
                - traffic_lights: Array of traffic light indicators (red/yellow/green)
        """
        gp_input_list = []
        for idx, sequence in enumerate(gp_data):
            # If sequence is 1-dimensional, reshape it to 2D (1, n_features)
            if sequence.ndim == 1:
                sequence = sequence.reshape(1, -1)
            # Flatten the sequence into a 1D array
            flattened = sequence.flatten()
            # Append the corresponding reconstruction error to the flattened sequence
            gp_input = np.append(flattened, gp_errors[idx])
            gp_input_list.append(gp_input)

        gp_input_array = np.stack(gp_input_list) # Stack all inputs into a single numpy array
        gp_input_tensor = torch.tensor(gp_input_array, dtype=torch.float64) # Convert numpy array to PyTorch tensor to make it ready for the neural network enabled at inference 

        try:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                gp_output = self.gp_model(gp_input_tensor) #Here we get GP model predictions --> This variable contains the predictions 
                uncertainties = gp_output.variance.detach().cpu().numpy()

            thresholds = self.analyze_uncertainties(uncertainties, self.gp_model, gp_input_tensor) # Calculate threshold values based on uncertainty distribution
            traffic_lights = self.assign_traffic_lights(uncertainties, thresholds) # Assign traffic light indicators based on uncertainties and thresholds

            return uncertainties, thresholds, traffic_lights

        except Exception as e:
            print(f"Error in GP model processing: {e}")
            print(f"GP input tensor shape: {gp_input_tensor.shape}")
            print(f"GP input tensor dtype: {gp_input_tensor.dtype}")
            raise

    def process_stacked_gp_data(self, sequence_data, residuals, timestamps):
        
        print(f"Processing stacked GP data with {len(sequence_data)} sequences")
        gp_input_list = []
        for idx, sequence in enumerate(sequence_data):
            # Reshape sequence to match training data structure
            mean_residual = np.mean(residuals[idx])  # Reshape to n and get the 2D array --> 18 features per timestep 

            flattened = sequence.flatten()  # 288 features
            gp_input = np.append(flattened, mean_residual)  # 288 + 1 = 289 features
            gp_input_list.append(gp_input)

        
        gp_input_tensor = torch.tensor(np.array(gp_input_list)).double()
        print(f"Final GP input tensor shape: {gp_input_tensor.shape}") 
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            gp_output = self.gp_model_stacked(gp_input_tensor)
            uncertainties = gp_output.variance.detach().cpu().numpy()
        # Process results
        thresholds = self.analyze_uncertainties(uncertainties, self.gp_model_stacked, gp_input_tensor)
        traffic_lights = self.assign_traffic_lights(uncertainties, thresholds)
        return uncertainties, thresholds, traffic_lights


    #This influx function creates     
    def push_stacked_gp_to_influx(self, timestamp, uncertainty, traffic_light, reconstruction_error, thresholds):
    
        # Convert reconstruction_error to scalar if it's an array
        if isinstance(reconstruction_error, (np.ndarray, list)):
            reconstruction_error = np.mean(reconstruction_error)
        
        gp_data_point = Point("StackedAE_GPPredictions")
        gp_data_point.tag("model", "stacked_autoencoder_gp")
        gp_data_point.time(timestamp, WritePrecision.NS)
        gp_data_point.field("uncertainty", float(uncertainty))
        gp_data_point.field("traffic_light", traffic_light)
        gp_data_point.field("traffic_light_value", {"green": 0, "yellow": 1, "red": 2}[traffic_light])
        gp_data_point.field("reconstruction_error", float(reconstruction_error))
        gp_data_point.field("low_threshold", float(thresholds['low_threshold']))
        gp_data_point.field("high_threshold", float(thresholds['high_threshold']))
        
        self.write_api.write(bucket=bucket, org=org, record=gp_data_point)
        
    

    def analyze_uncertainties(self, uncertainties, gp_model, gp_input):
        # Calculate basic measures of uncertainty 
        mean_uncertainty = np.mean(uncertainties)
        std_uncertainty = np.std(uncertainties)
        
    #The following print blocks can optionally be used for more detailed analysis of change in variance of the distributions

        #print(f"Mean uncertainty: {mean_uncertainty:.6f}")
        #print(f"Standard deviation of uncertainty: {std_uncertainty:.6f}")
        #print(f"Min uncertainty: {np.min(uncertainties):.6f}")
        #print(f"Max uncertainty: {np.max(uncertainties):.6f}")

        # Calculate ELBO (Evidence Lower BOund) as a proxy for information gain
        gp_model.eval()
        with torch.no_grad():
            output = gp_model(gp_input)
            # Get variational distribution from model
            variational_dist = gp_model.variational_strategy.variational_distribution
            kl_divergence = gp_model.variational_strategy.kl_divergence().item()
            # Calculate log likelihood of the data
            log_likelihood = output.log_prob(gp_input[:, -1]).mean().item() 
            elbo = log_likelihood - kl_divergence   # ELBO = log likelihood - KL divergence
            # Higher ELBO indicates better model fit for the data

        # Define thresholds based on ELBO
        low_threshold = mean_uncertainty - 0.5 * std_uncertainty
        high_threshold = mean_uncertainty + 0.5 * std_uncertainty

        # Return thresholds and metrics from the dictionary  
        return {
            'mean': mean_uncertainty,
            'mean_plus_std': mean_uncertainty + std_uncertainty,
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
            'elbo': elbo,
            'kl_divergence': kl_divergence,
            'log_likelihood': log_likelihood,
        }


    def assign_traffic_lights(self, uncertainties, thresholds):
        traffic_lights = []
        for uncertainty in uncertainties:
            if uncertainty < thresholds['low_threshold']:
                traffic_lights.append('red')
            elif uncertainty < thresholds['high_threshold']:
                traffic_lights.append('yellow')
            else:
                traffic_lights.append('green')
        
        return traffic_lights

    def push_gp_data_to_influx(self, buffer_index, timestamp, uncertainty, traffic_light, reconstruction_error, thresholds):
        gp_data_point = Point("GPPredictions")
        gp_data_point.tag("buffer_index", buffer_index)
        gp_data_point.time(timestamp, WritePrecision.NS)
        gp_data_point.field("uncertainty", float(uncertainty))
        gp_data_point.field("traffic_light_value", {"green": 0, "yellow": 1, "red": 2}[traffic_light])
        gp_data_point.field("reconstruction_error", float(reconstruction_error))
        gp_data_point.field("low_threshold", float(thresholds['low_threshold']))
        gp_data_point.field("high_threshold", float(thresholds['high_threshold']))
        self.write_api.write(bucket=bucket, org=org, record=gp_data_point)
    
    def save_buffer_to_csv(self, buffer, buffer_index):
        df = pd.DataFrame(buffer)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"buffer_{buffer_index}_{timestamp}.csv"
        
        # Save to CSV
        filepath = os.path.join(self.csv_storage, filename)
        df.to_csv(filepath, index=False)
        #print(f"Buffer {buffer_index} saved to {filepath}")
    
    
    def load_buffer_from_csv(self, buffer_index):
        # Determine the file pattern based on the buffer_index
        if buffer_index == "stacked_ae":
            file_pattern = "buffer_stacked_ae_*.csv"
        else:
            file_pattern = f"buffer_Ultra_*.csv"

        # Get the most recent CSV file for the given buffer index
        csv_files = glob(os.path.join(self.csv_storage, file_pattern))
        if not csv_files:
            print(f"No CSV files found for buffer {buffer_index}")
            return None

        latest_file = max(csv_files, key=os.path.getctime)
        
        # Read the CSV file
        df = pd.read_csv(latest_file, header=0)
        # Convert DataFrame back to list of lists
        buffer_data = df.values.tolist()
        return buffer_data


def get_server_time(client, retries=3, timeout=10):
    server_time_node = client.get_node("ns=0;i=2258")
    for attempt in range(retries):
        try:
            return server_time_node.get_value()
        except TimeoutError:
            if attempt < retries - 1:
                print(f"TimeoutError: Retrying to get server time (attempt {attempt + 1}/{retries})")
                time.sleep(timeout)
            else:
                print("Failed to get server time after multiple attempts, using local system time")
                return datetime.datetime.utcnow()

def cleanup(client):
    print("Entering Cleanup Phase")
    try:
        client.close()
    except TypeError:
        pass

def write_to_influx(write_api, series_name, fields, tags=None, timestamp=None): 
    try:
        point = Point(series_name)
        for field_key, field_value in fields.items():
            point = point.field(field_key, field_value)
        if tags:
            for tag_key, tag_value in tags.items():
                point = point.tag(tag_key, tag_value)
        if timestamp:
            point = point.time(timestamp, WritePrecision.NS)
        write_api.write(bucket=bucket, org=org, record=point)
    except Exception as e:
        print(f"Error writing to InfluxDB: {e}")
        print(f"Attempted to write: Series={series_name}, Fields={fields}, Tags={tags}, Timestamp={timestamp}")

'''Creates sequences to iterate through datapoints and adds the data to a tensor format so it can be fed to a neural network for processing'''
def create_sequences(buffer, sequence_length, step_size):
    sequences = []
    for i in range(0, len(buffer) - sequence_length + 1, int(step_size)):
        sequence = buffer[i:i+sequence_length]
        sequences.append(torch.tensor(sequence, dtype=torch.float64))
    return torch.stack(sequences)



'''
For the rest of the code below scenarios with different configurable parameters can be applied as well as OPC client initialization, connection to the server and specifics around
bucket use, and other key details to get to the right influx server, as specified in the configuration file 
'''
opc_client = Client(opc_client_url, timeout=15)
opc_client.connect()
nodeids = ["ns=1;s=0xA0"]
nodes = [opc_client.get_node(nodeid) for nodeid in nodeids]
print(nodes)

token = configuration['influxdb_token'] #Collect the password token from the configuration file, used for InfluxDB. Org, Url, bucket, thresholds are values found in the configuration file
org = configuration['influxdb_org'] 
url = configuration['influxdb_url']
bucket = configuration['influxdb_bucket']
thresholds = configuration['thresholds']

client = influxdb_client.InfluxDBClient(url=url, token=token, org=org) #we call on a client to set up a new connection. 
atexit.register(cleanup, client)
write_api = client.write_api(write_options=SYNCHRONOUS)

scenarios = [{"batch_size": 10, "sequence_length": 16}, #scenarios control the different parameterizable configurations for two hyperparameters with significant impact on performance
    {"batch_size": 32, "sequence_length": 32},
    {"batch_size": 64, "sequence_length": 64}
]

email_counter = 0
email_interval = 12000 #How often an email with the log file should be sent (counted in number of full iterations through the main loop) 

gp_model, gp_model_stacked = load_gp_model(configuration['TCNATT-GP_model_path'], configuration['StackedAE-GP_model_path'])

code_logic = CodeLogic(write_api, opc_client, gp_model, gp_model_stacked, thresholds)


try:
    while True:
        try:
            code_logic.fetch_data()
            time.sleep(1)  
            
            
            email_counter += 1
            if email_counter >= email_interval:
                try:
                    send_log_email(log_filename, "stellanlange@gmail.com")
                    email_counter = 0
                except Exception as e:
                    print(f"Error sending email: {e}")
                    
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
            traceback.print_exc()
            time.sleep(5)  #wait a little before retrying

except KeyboardInterrupt:
    print("Interrupted by user. Exiting...") 

except Exception as e:
    print(f"An error occurred: {e}") 

finally:
    try:
        client.close()
    except TypeError:
        pass
    opc_client.disconnect() # We finally make a full disconnect from the server as we enter the cleanup phase
        
    
    #dev.close()
    if isinstance(sys.stdout, TeeLogger): #Here we are using teelogger to simultenously (not fully) write terminal output to the terminal and a log file, and it includes error and standard operations
        sys.stdout.log.close()
        sys.stdout = sys.__stdout__
    if isinstance(sys.stderr, TeeLogger):
        sys.stderr.log.close()
        sys.stderr = sys.__stderr__