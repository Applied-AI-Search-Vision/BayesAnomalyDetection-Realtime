import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, SpectralMixtureKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor
from torch.nn.utils import parametrizations as param
import torch.linalg as LA


def matrix_sqrt(matrix):
    """Compute the square root of a positive definite matrix."""
    _, s, v = matrix.svd()
    good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)

gp_sequence_length = 32
gp_buffer_size =  128
device = torch.device("cpu")

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        
        x = self.conv(x)
        
        x = x[:, :, :x.size(2) - self.padding]
        
        return x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        # First convolution layer
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation)
        self.conv1.conv = param.weight_norm(self.conv1.conv)
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution layer
        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation)
        self.conv2.conv = param.weight_norm(self.conv2.conv)
        self.dropout2 = nn.Dropout(dropout)

        # Downsample layer for residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()


    def init_weights(self):
        self.conv1.conv.weight.data.normal_(0, 0.01)
        self.conv2.conv.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        
        
        res = x if self.downsample is None else self.downsample(x)
        

        out = self.relu(self.conv1(x))
        

        out = self.relu(self.conv2(out))
        out = self.dropout2(out)
        

        out = self.relu(out + res)
        
        return out
    

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        
        for i, layer in enumerate(self.network):
            x = layer(x)
            
            
        return x

class SelfAttention(nn.Module): 
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        
        # Process in smaller chunks if input is too large
        if N > 64:  # Adjust this threshold based on your memory
            chunk_size = 8
            attention_chunks = []
            attention_weights_chunks = []
            
            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                chunk_values = values[i:end_idx]
                chunk_keys = keys[i:end_idx]
                chunk_queries = query[i:end_idx]
                chunk_mask = mask[i:end_idx] if mask is not None else None
                
                # Process chunk
                chunk_out, chunk_weights = self._process_chunk(
                    chunk_values, chunk_keys, chunk_queries, chunk_mask
                )
                attention_chunks.append(chunk_out)
                attention_weights_chunks.append(chunk_weights)
            
            return (torch.cat(attention_chunks, dim=0), 
                   torch.cat(attention_weights_chunks, dim=0))
        
        return self._process_chunk(values, keys, query, mask)

    def _process_chunk(self, values, keys, query, mask):
        N = query.shape[0]
        values = values.reshape(N, -1, self.heads, self.head_dim)
        keys = keys.reshape(N, -1, self.heads, self.head_dim)
        queries = query.reshape(N, -1, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Free memory explicitly
        torch.cuda.empty_cache()

        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))
        
        attention_weights = torch.softmax(
            attention / (self.embed_size ** (1/2)), dim=3
        )
        
        out = torch.einsum(
            "nhql,nlhd->nqhd", 
            [attention_weights, values]
        ).reshape(N, -1, self.heads * self.head_dim)
        
        out = self.fc_out(out)
        return out, attention_weights

class TCNWithAttention(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, embed_size, heads):
        super().__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout) #The first TCN-Attention handshake
        self.attention = SelfAttention(embed_size, heads)

        self.fc = nn.Linear(embed_size, num_inputs)

    def forward(self, x, mask = None, return_latent=False):
        tcn_out = self.tcn(x)
        tcn_out_flat = tcn_out.view(tcn_out.size(0), -1)
        attn_out, attn_weights = self.attention(tcn_out_flat, tcn_out_flat, tcn_out_flat, mask)
        attn_out = attn_out.view(tcn_out.size(0), tcn_out.size(2), -1)
        out = self.fc(attn_out)

        return out, attn_weights

# Initialize the model
num_inputs = 12
num_channels = [12, 64, 128]  
kernel_size = 3
dropout = 0.2827
embed_size = 128
heads = 4
batch_size = 16
sequence_length = 128
step_size = 1


def buffer_generator(csv_file, buffer_size=32, stride=32):
    data = pd.read_csv(csv_file)
    normalized_data = (data - data.mean()) / data.std()
    
    for i in range(0, len(normalized_data) - buffer_size + 1, stride):
        buffer = normalized_data.iloc[i:i+buffer_size].values
        timestamp = data.index[i+buffer_size-1]  # Use the last timestamp of the buffer
        yield buffer, timestamp

class AutocovarianceKernel(gpytorch.kernels.Kernel):
    def __init__(self, acovf, batch_shape=torch.Size(), active_dims=None):
        super().__init__(batch_shape=batch_shape, active_dims=active_dims)
        self.register_parameter(name="raw_acovf", parameter=torch.nn.Parameter(torch.tensor(acovf)))

    def forward(self, x1, x2, diag=False, **params):
        acovf = self.raw_acovf.exp()
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        lags = diff.abs().clamp(max=len(acovf)-1)
        covar = acovf[lags.long()]
        
        if diag:
            return covar.diagonal(dim1=-1, dim2=-2)
        else:
            return covar

class SpectralMixtureGPModel(ApproximateGP):
    def __init__(self, inducing_points, likelihood, num_mixtures=4, acovf=None):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SpectralMixtureGPModel, self).__init__(variational_strategy)
        
        self.mean_module = ConstantMean()
        kernels = [
            RBFKernel(ard_num_dims=inducing_points.size(1)),
            SpectralMixtureKernel(num_mixtures=num_mixtures, ard_num_dims=inducing_points.size(1))
        ]
        if acovf is not None:
            kernels.append(AutocovarianceKernel(acovf))
        self.covar_module = ScaleKernel(sum(kernels))
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        # Use matrix_sqrt for more stable computations
        Kmm = matrix_sqrt(covar_x)
        Knn = matrix_sqrt(self.covar_module(x, x))
        Kmn = self.covar_module(self.variational_strategy.inducing_points, x)
        
        noise = self.likelihood.noise
        
        # Add small jitter to the diagonal for numerical stability
        jitter = 1e-6 * torch.eye(Kmm.shape[-1], device=Kmm.device)
        Kmm += jitter
        Knn += jitter
        
        Q = Kmn.transpose(-1, -2) @ Kmm.inv_matmul(Kmn)
        G = DiagLazyTensor((Knn - Q).diag() + noise.unsqueeze(-1))
        
        B = Kmm + G.inv_matmul(Kmn.transpose(-1, -2).evaluate(), left_tensor=Kmn.evaluate())
        
        return MultivariateNormal(mean_x, B)

    def predict(self, x):
        self.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self(x)
            mean = observed_pred.mean
            covar = observed_pred.covariance_matrix
            
            # Add small jitter to the diagonal for numerical stability
            jitter = 1e-6 * torch.eye(covar.shape[-1], device=covar.device)
            covar += jitter
            
            return mean, covar

# KL Divergence calculation
def kl_divergence_mvn(p, q):
    mean_diff = q.mean - p.mean
    cov_p, cov_q = p.covariance_matrix, q.covariance_matrix
    
    k = p.event_shape[0]  # Dimensionality
    
    jitter = 1e-6 * torch.eye(k, device=cov_p.device)
    cov_p += jitter
    cov_q += jitter

    term1 = torch.logdet(cov_q) - torch.logdet(cov_p)
    term2 = torch.trace(torch.linalg.inv(cov_q) @ cov_p)
    term3 = mean_diff @ torch.linalg.inv(cov_q) @ mean_diff
    
    return 0.5 * (term1 - k + term2 + term3)

# Uncertainty quantification using GP
def quantify_uncertainty_gp(gp_model, normalized_anomalies):
    gp_model.eval()
    uncertainties = []
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for sequence in normalized_anomalies:
            x = sequence.view(1, -1)
            try:
                prior_dist = gp_model.prior(x)
                posterior_dist = gp_model(x)
                kl_div = kl_divergence_mvn(posterior_dist, prior_dist)
                uncertainties.append(kl_div.item())
            except Exception as e:
                print(f"Error in GP_Inference: {e}")
                uncertainties.append(float('nan'))

    
    return uncertainties


def init_model():
    print(f'init_model start')
    model = TCNWithAttention(num_inputs=num_inputs, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout, embed_size=embed_size, heads=heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.6835084022669533e-05)



    # Load the trained model weights
    checkpoint_path = 'C:\\Users\\Stell\\Desktop\\Traffic Lights UQ\\TCNATT_checkpoint_epoch_15_non_duplicates.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval() 

    return model

def loadData(file_paths: list, timesteps: int, four_features) -> tuple:
    df_list = []
    timestamp_list = []  
    
    for path in file_paths:
        df = pd.read_csv(path, skiprows=3)

        # Extract relevant columns
        machine_id_col = 'Link'
        axis_col = '_field'
        value_col = '_value'
        time_col = '_time'

        

        
        filtered_df = df[[machine_id_col, axis_col, value_col, time_col]]



        
        pivoted_df = filtered_df.pivot_table(index=[machine_id_col, time_col], columns=[axis_col], values=value_col)

        
        pivoted_df.reset_index(inplace=True)

        
        pivoted_df[time_col] = pd.to_datetime(pivoted_df[time_col])

        
        pivoted_df.sort_values(by=[machine_id_col, time_col], inplace=True)

        
        pivoted_df.ffill(inplace=True)  
        pivoted_df.dropna(inplace=True)  
        
        
        timestamps = pivoted_df[time_col]

        
        if four_features:
        
            first_four_features = [f'Axis Z value{i}' for i in range(4)]
            feature_df = pivoted_df[first_four_features]
        else:
            features = ['Axis X', 'Axis Y', 'Axis Z']
            twelve_features = [f'{axis} value{i}' for axis in features for i in range(4)]
            feature_df = pivoted_df[twelve_features]


        
        df_list.append(feature_df)

        
        timestamp_list.append(timestamps)


    
    concatenated_time_series = np.concatenate([df.values for df in df_list])

    
    concatenated_timestamps = pd.concat(timestamp_list).reset_index(drop=True).values

 
    
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(concatenated_time_series)
    
    # Final sanity check: Ensure the number of timestamps matches the number of data points
    if len(concatenated_timestamps) != len(normalized_data):
        raise ValueError("The length of timestamps does not match the length of the normalized data.")

    normalized_tensor = torch.FloatTensor(normalized_data)

    

    return normalized_tensor, normalized_data, concatenated_timestamps



def create_sequences(data, sequence_length, step_size):
    
    sequences = []
    for i in range(0, len(data) - sequence_length+(sequence_length)+1, int(step_size)):
        sequence = data[i:i+sequence_length]
        if len(sequence) < sequence_length: 
            break

        sequences.append(sequence)
        
        
    return torch.stack(sequences)



def calculate_reconstruction_errors(model, data_loader):
    print("Calculating reconstruction errors")
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for sequences in data_loader:
            sequences = sequences.to(device)
            reconstructed, attn_weights = model(sequences, mask=None)
            reconstructed = torch.transpose(reconstructed, 1, 2)
            error = ((sequences - reconstructed) ** 2).mean(dim=(1, 2))
            reconstruction_errors.extend(error.cpu().numpy())
    return np.array(reconstruction_errors)


def detect_anomalies(reconstruction_errors, threshold):
    print("Detecting anomalies")
    high_error_indices = np.where(reconstruction_errors > threshold)[0]
    return high_error_indices

class HODLRMatrix:
    def __init__(self, matrix, rank=10, min_size=64):
        self.min_size = min_size
        self.rank = rank
        # Convert LinearOperator to dense tensor if needed
        self.matrix = matrix.to_dense() if hasattr(matrix, 'to_dense') else matrix
        self.size = self.matrix.shape[0]
        self.is_leaf = self.size <= min_size
        
        if not self.is_leaf:
            self.mid = self.size // 2
            
            # Split matrix into blocks
            self.A11 = self.matrix[:self.mid, :self.mid]
            self.A22 = self.matrix[self.mid:, self.mid:]
            
            # Convert to dense and compute SVD
            off_diag1 = self.matrix[:self.mid, self.mid:].to_dense() if hasattr(self.matrix, 'to_dense') else self.matrix[:self.mid, self.mid:]
            off_diag2 = self.matrix[self.mid:, :self.mid].to_dense() if hasattr(self.matrix, 'to_dense') else self.matrix[self.mid:, :self.mid]
            
            # Compute low-rank approximations using torch.linalg.svd
            try:
                U1, S1, V1 = torch.linalg.svd(off_diag1, full_matrices=False)
                U2, S2, V2 = torch.linalg.svd(off_diag2, full_matrices=False)
                
                # Keep only top 'rank' singular values/vectors
                rank = min(rank, min(U1.size(1), U2.size(1)))
                
                self.U1 = U1[:, :rank] * torch.sqrt(S1[:rank].unsqueeze(0))
                self.V1 = V1[:rank, :].t() * torch.sqrt(S1[:rank].unsqueeze(0))
                self.U2 = U2[:, :rank] * torch.sqrt(S2[:rank].unsqueeze(0))
                self.V2 = V2[:rank, :].t() * torch.sqrt(S2[:rank].unsqueeze(0))
                
                # Recursively construct HODLR for diagonal blocks
                self.hodlr11 = HODLRMatrix(self.A11, rank, min_size)
                self.hodlr22 = HODLRMatrix(self.A22, rank, min_size)
            
            except RuntimeError as e:
                print(f"SVD failed: {e}")
                # Fallback to identity matrices if SVD fails
                self.is_leaf = True
                self.matrix = torch.eye(self.size, device=self.matrix.device)

class HODLRKernel(gpytorch.kernels.Kernel):
    def __init__(self, base_kernel, rank=10, min_size=64, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.rank = rank
        self.min_size = min_size

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return self.base_kernel(x1, x2, diag=True, **params)
        
        # Compute base kernel matrix
        K = self.base_kernel(x1, x2, **params)
        
        # Convert to dense if needed
        if hasattr(K, 'to_dense'):
            K = K.to_dense()
        
        # Create HODLR approximation
        try:
            hodlr_K = HODLRMatrix(K, self.rank, self.min_size)
            return hodlr_K.matrix  # Return the approximated matrix
        except Exception as e:
            print(f"HODLR approximation failed: {e}")
            return K  # Fallback to original matrix

class VariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, rank=10, min_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(VariationalGPModel, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Create base kernel with proper structure
        self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(1))
        self.matern_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=inducing_points.size(1))
        
        # Combine kernels
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.base_kernel + self.matern_kernel
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def efficient_matrix_operations(hodlr_matrix, vector):
    """Efficient matrix-vector multiplication using HODLR structure"""
    if hodlr_matrix.is_leaf:
        return torch.mv(hodlr_matrix.matrix, vector)
    
    v1 = vector[:hodlr_matrix.mid]
    v2 = vector[hodlr_matrix.mid:]
    
    # Multiply diagonal blocks
    result1 = efficient_matrix_operations(hodlr_matrix.hodlr11, v1)
    result2 = efficient_matrix_operations(hodlr_matrix.hodlr22, v2)
    
    # Add off-diagonal contributions
    result1 += torch.mv(hodlr_matrix.U1, torch.mv(hodlr_matrix.V1.t(), v2))
    result2 += torch.mv(hodlr_matrix.U2, torch.mv(hodlr_matrix.V2.t(), v1))
    
    return torch.cat([result1, result2])

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_tcnatt_anomalies(original_data, reconstructed_data, anomaly_indices, test_timestamps, downsample_factor=1):
    # Ensure all inputs are numpy arrays
    original_data = np.array(original_data)
    reconstructed_data = np.array(reconstructed_data)
    test_timestamps = np.array(test_timestamps)

    # Downsample the data
    original_data = original_data[::downsample_factor]
    reconstructed_data = reconstructed_data[::downsample_factor]
    test_timestamps = test_timestamps[::downsample_factor]

    # Ensure all arrays have the same length
    min_length = min(len(original_data), len(reconstructed_data), len(test_timestamps))
    original_data = original_data[:min_length]
    reconstructed_data = reconstructed_data[:min_length]
    test_timestamps = test_timestamps[:min_length]

    n_features = original_data.shape[1]
    anomaly_mask_test = np.zeros(len(original_data), dtype=bool)
    anomaly_mask_test[anomaly_indices] = True

    fig = make_subplots(rows=4, cols=3, subplot_titles=[f'Feature {i+1}' for i in range(n_features)])

    for i in range(n_features):
        row = i // 3 + 1
        col = i % 3 + 1

        # Original data
        fig.add_trace(go.Scatter(x=test_timestamps, y=original_data[:, i], mode='lines', name=f'Original Feature {i+1}', line=dict(color='blue')), row=row, col=col)
        
        # Reconstructed data
        fig.add_trace(go.Scatter(x=test_timestamps, y=reconstructed_data[:, i], mode='lines', name=f'Reconstructed Feature {i+1}', line=dict(color='orange')), row=row, col=col)

        # Anomalies
        fig.add_trace(go.Scatter(
            x=test_timestamps[anomaly_mask_test], y=original_data[anomaly_mask_test, i],
            mode='markers', name='Anomalies',
            marker=dict(color='red', size=5),
            showlegend=i==0
        ), row=row, col=col)

    fig.update_layout(height=1200, width=1600, title_text="TCN-ATT Model Anomaly Detection")
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Value")

    # Save the plot as an HTML file
    fig.write_html("tcnatt_anomaly_plot_interactive.html")
    print("Interactive TCN-ATT anomaly plot saved as 'tcnatt_anomaly_plot_interactive.html'")

# Define assign_traffic_lights function
def assign_traffic_lights(uncertainties, thresholds):
    traffic_lights = []
    for uncertainty in uncertainties:
        if uncertainty < thresholds['low_threshold']:
            traffic_lights.append('green')
        elif uncertainty < thresholds['high_threshold']:
            traffic_lights.append('yellow')
        else:
            traffic_lights.append('red')
    
    return traffic_lights

def plot_anomalies_and_predictions_plotly(original_data, reconstructed_data, anomaly_indices, traffic_lights, timestamps):
    print("\nPlotting data...")
    print(f"Data shapes:")
    print(f"Original data: {len(original_data)}")
    print(f"Reconstructed data: {len(reconstructed_data)}")
    print(f"Traffic lights: {len(traffic_lights)}")
    print(f"Timestamps: {len(timestamps)}")
    print(f"Anomaly indices: {len(anomaly_indices)}")

    # Create figure
    fig = go.Figure()

    # Plot original data
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=original_data,
        mode='lines',
        name='Original Data',
        line=dict(color='blue', width=1)
    ))

    # Create color mapping for traffic lights
    color_map = {
        'red': 'rgb(255,0,0)',
        'yellow': 'rgb(255,255,0)',
        'green': 'rgb(0,255,0)'
    }

    # Plot points with their respective traffic light colors
    for color in ['red', 'yellow', 'green']:
        mask = [light == color for light in traffic_lights]
        if any(mask):
            fig.add_trace(go.Scatter(
                x=[timestamps[i] for i in range(len(mask)) if mask[i]],
                y=[original_data[i] for i in range(len(mask)) if mask[i]],
                mode='markers',
                name=f'{color.capitalize()} Points',
                marker=dict(
                    color=color_map[color],
                    size=8,
                    opacity=0.6
                )
            ))

    # Highlight anomalies
    if len(anomaly_indices) > 0:
        fig.add_trace(go.Scatter(
            x=[timestamps[i] for i in anomaly_indices if i < len(timestamps)],
            y=[original_data[i] for i in anomaly_indices if i < len(original_data)],
            mode='markers',
            name='Anomalies',
            marker=dict(
                color='red',
                size=12,
                line=dict(
                    color='black',
                    width=2
                )
            )
        ))

    # Update layout
    fig.update_layout(
        title='Anomaly Detection with Traffic Light System',
        xaxis_title='Timestamp',
        yaxis_title='Value',
        showlegend=True,
        legend_title='Legend',
        hovermode='closest'
    )

    # Save the plot
    fig.write_html("anomaly_detection_plot.html")
    print("Plot saved as 'anomaly_detection_plot.html'")

def analyze_uncertainties(uncertainties, gp_model, gp_input):
    mean_uncertainty = np.mean(uncertainties)
    std_uncertainty = np.std(uncertainties)
    
    print(f"Mean uncertainty: {mean_uncertainty:.6f}")
    print(f"Standard deviation of uncertainty: {std_uncertainty:.6f}")
    print(f"Min uncertainty: {np.min(uncertainties):.6f}")
    print(f"Max uncertainty: {np.max(uncertainties):.6f}")
    
    # Plot histogram
    #plt.figure(figsize=(10, 6))
    #plt.hist(uncertainties, bins=50, edgecolor='black')
    #plt.title('Distribution of Uncertainties')
    #plt.xlabel('Uncertainty')
    #plt.ylabel('Frequency')
    #plt.axvline(mean_uncertainty, color='r', linestyle='dashed', linewidth=2, label='Mean')
    #plt.axvline(mean_uncertainty + std_uncertainty, color='g', linestyle='dashed', linewidth=2, label='Mean + 1 Std Dev')
    #plt.legend()
    #plt.show()

    # Calculate ELBO (Evidence Lower BOund) as a proxy for information gain
    gp_model.eval()
    with torch.no_grad():
        output = gp_model(gp_input)
        variational_dist = gp_model.variational_strategy.variational_distribution
        kl_divergence = gp_model.variational_strategy.kl_divergence().item()
        log_likelihood = output.log_prob(gp_input[:, -1]).mean().item()
        elbo = log_likelihood - kl_divergence

    # Define thresholds based on ELBO
    low_threshold = mean_uncertainty - 0.5 * std_uncertainty
    high_threshold = mean_uncertainty + 0.5 * std_uncertainty

    print(f"ELBO: {elbo:.6f}")
    print(f"KL Divergence: {kl_divergence:.6f}")
    print(f"Log Likelihood: {log_likelihood:.6f}")
    print(f"Low Threshold: {low_threshold:.6f}")
    print(f"High Threshold: {high_threshold:.6f}")

    # Return meaningful thresholds for further analysis
    return {
        'mean': mean_uncertainty,
        'mean_plus_std': mean_uncertainty + std_uncertainty,
        'low_threshold': low_threshold,
        'high_threshold': high_threshold,
        'elbo': elbo,
        'kl_divergence': kl_divergence,
        'log_likelihood': log_likelihood,
    }

def prepare_gp_input_inference(sequences, errors, anomaly_indices, gp_sequence_length, buffer_size, 
                             real_time_emulation=True, step_size=1):
    gp_inference_inputs = []
    
    # Convert errors to torch.Tensor if it's a NumPy array
    if isinstance(errors, np.ndarray):
        errors = torch.from_numpy(errors).float()
    
    # Move errors to the same device as sequences
    errors = errors.to(sequences.device)
    
    # Select only anomalous sequences and errors
    anomalous_sequences = sequences[anomaly_indices]  # Shape: [num_anomalies, 12, 128]
    anomalous_errors = errors[anomaly_indices]       # Shape: [num_anomalies, 1]
    
    # Calculate correct dimensions
    batch_dim = gp_sequence_length
    feature_dim = anomalous_sequences.shape[1] * anomalous_sequences.shape[2]  # 12 * 128 = 1536
    
    print(f"Debug - Shapes:")
    print(f"Anomalous sequences shape: {anomalous_sequences.shape}")
    print(f"Anomalous errors shape: {anomalous_errors.shape}")
    print(f"Feature dimension: {feature_dim}")
    
    if real_time_emulation:
        # Real-time emulation with sliding window
        for i in range(0, len(anomalous_sequences) - gp_sequence_length + 1, step_size):
            try:
                # Get window of sequences
                window_sequences = anomalous_sequences[i:i+gp_sequence_length]  # [32, 12, 128]
                window_errors = anomalous_errors[i:i+gp_sequence_length]        # [32, 1]
                
                # Reshape window_sequences correctly
                window_sequences_flat = window_sequences.reshape(1, -1)  # [1, 32*12*128]
                window_sequences_flat = window_sequences_flat[:, :feature_dim]  # [1, 1536]
                
                # Reshape errors
                window_errors_flat = window_errors.mean().reshape(1, 1)  # Take mean of errors in window
                
                # Concatenate
                gp_input = torch.cat([window_sequences_flat, window_errors_flat], dim=1)
                gp_inference_inputs.append(gp_input)
                
            except Exception as e:
                print(f"Error processing window {i}: {e}")
                print(f"Window sequences shape: {window_sequences.shape}")
                continue
    else:
        # Batch processing with buffer
        for i in range(0, len(anomalous_sequences), buffer_size):
            try:
                # Get buffer of sequences
                buffer_sequences = anomalous_sequences[i:i+buffer_size]
                buffer_errors = anomalous_errors[i:i+buffer_size]
                
                # Handle incomplete buffers
                if len(buffer_sequences) < buffer_size:
                    padding_size = buffer_size - len(buffer_sequences)
                    padding_sequence = buffer_sequences[-1:].repeat(padding_size, 1, 1)
                    padding_error = buffer_errors[-1:].repeat(padding_size)
                    
                    buffer_sequences = torch.cat([buffer_sequences, padding_sequence], dim=0)
                    buffer_errors = torch.cat([buffer_errors, padding_error], dim=0)
                
                # Reshape correctly
                buffer_sequences_flat = buffer_sequences.reshape(buffer_size, -1)  # [buffer_size, 12*128]
                buffer_sequences_flat = buffer_sequences_flat[:, :feature_dim]     # [buffer_size, 1536]
                buffer_errors_flat = buffer_errors.reshape(buffer_size, 1)         # [buffer_size, 1]
                
                # Concatenate
                gp_input = torch.cat([buffer_sequences_flat, buffer_errors_flat], dim=1)
                gp_inference_inputs.append(gp_input)
                
            except Exception as e:
                print(f"Error processing buffer {i}: {e}")
                print(f"Buffer sequences shape: {buffer_sequences.shape}")
                continue

    # Print shapes for debugging
    if len(gp_inference_inputs) > 0:
        print(f"GP inference input shape: {gp_inference_inputs[0].shape}")
        print(f"Expected shape: [batch_size, {feature_dim + 1}]")  # 1536 + 1 = 1537
    
    return gp_inference_inputs

def plot_results_with_traffic_lights(normalized_test_data, test_timestamps, anomaly_indices, uncertainties, traffic_lights):
    fig = go.Figure()

    # Create a full uncertainties array
    full_uncertainties = np.zeros(len(normalized_test_data))
    full_uncertainties[anomaly_indices] = uncertainties

    # Plot the original data
    fig.add_trace(go.Scatter(
        x=test_timestamps, 
        y=normalized_test_data,
        mode='lines', 
        name='Original Data', 
        line=dict(color='blue')
    ))

    # Create a color map for traffic lights
    color_map = {'red': 'rgb(255,0,0)', 'yellow': 'rgb(255,255,0)', 'green': 'rgb(0,255,0)'}

    # Plot points for each traffic light color
    for color in ['red', 'yellow', 'green']:
        # Create mask for this color
        color_mask = np.array([light == color for light in traffic_lights])
        
        if np.any(color_mask):
            # Get indices where this color appears
            color_indices = np.where(color_mask)[0]
            
            # Create hover text
            hover_text = []
            for idx in color_indices:
                if idx in anomaly_indices:
                    # Find the position in anomaly_indices to get the correct uncertainty value
                    uncertainty_idx = np.where(anomaly_indices == idx)[0][0]
                    hover_text.append(f"Uncertainty: {uncertainties[uncertainty_idx]:.6f}")
                else:
                    hover_text.append("No uncertainty value")

            fig.add_trace(go.Scatter(
                x=test_timestamps[color_indices],
                y=normalized_test_data[color_indices],
                mode='markers',
                name=f'{color.capitalize()} Points',
                marker=dict(
                    color=color_map[color],
                    size=8,
                    opacity=0.6
                ),
                hovertext=hover_text,
                hoverinfo="text"
            ))

    # Update layout
    fig.update_layout(
        title='Time Series with Traffic Light System and Uncertainties',
        xaxis_title='Timestamp',
        yaxis_title='Normalized Value',
        showlegend=True,
        legend_title='Legend',
        hovermode='closest'
    )

    # Save the plot
    fig.write_html("traffic_lights_uncertainty_plot.html")
    print("\nPlot saved as 'traffic_lights_uncertainty_plot.html'")

def plot_gp_3d_distribution(gp_model, gp_input, uncertainties, title="GP Multivariate Distribution"):
    """
    Creates a 3D visualization of the GP's multivariate distribution
    """
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA

    # Perform PCA to reduce dimensionality to 3D for visualization
    pca = PCA(n_components=3)
    
    # Determine device
    device = next(gp_model.parameters()).device
    
    # Get the mean predictions and uncertainties
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        gp_input = gp_input.to(device)
        output = gp_model(gp_input)
        mean = output.mean.cpu().numpy()
        variance = output.variance.cpu().numpy()

    # Combine features for PCA
    combined_features = np.column_stack([
        gp_input.cpu().numpy(),
        mean.reshape(-1, 1),
        variance.reshape(-1, 1)
    ])
    
    # Apply PCA
    pca_result = pca.fit_transform(combined_features)
    
    # Create the 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            z=pca_result[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=uncertainties,  # Color points by uncertainty
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Uncertainty")
            ),
            hovertext=[f"Uncertainty: {u:.4f}<br>Mean: {m:.4f}<br>Variance: {v:.4f}"
                      for u, m, v in zip(uncertainties, mean, variance)],
            hoverinfo="text"
        )
    ])

    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            zaxis_title="PCA Component 3",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=30)
    )

    
    explained_var = pca.explained_variance_ratio_
    fig.add_annotation(
        text=f"Explained variance ratios:<br>PC1: {explained_var[0]:.3f}<br>PC2: {explained_var[1]:.3f}<br>PC3: {explained_var[2]:.3f}",
        xref="paper", yref="paper",
        x=0, y=1,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )

    fig.show()
    fig.write_html("gp_distribution_3d.html")





# Main function
def main(test_data_path):
    # Define hyperparameters at the start
    batch_size = 64
    num_epochs = 500
    learning_rate = 0.01
    weight_decay = 1e-4
    real_time_emulation = False
    
    # Step 1: Load and preprocess data
    test_csv_file = [test_data_path]
    normalized_test_tensor, normalized_test_data, test_timestamps = loadData(test_csv_file, 0, False)

    # Step 2: Anomaly detection with TCNATT
    tcnatt_model = init_model()
    tcnatt_model = tcnatt_model.to(device)
    tcnatt_model.eval()

    test_sequences = create_sequences(normalized_test_tensor, sequence_length, step_size)
    test_sequences = torch.transpose(test_sequences, 1, 2)
    test_loader = DataLoader(test_sequences, shuffle=False, batch_size=batch_size, drop_last=False)

    reconstruction_errors = calculate_reconstruction_errors(tcnatt_model, test_loader)
    threshold = np.percentile(reconstruction_errors, 98)
    anomaly_indices = detect_anomalies(reconstruction_errors, threshold)
    
    print(f"Number of anomalies detected: {len(anomaly_indices)}")
    print(f"Max anomaly index: {max(anomaly_indices) if anomaly_indices.size > 0 else 'N/A'}")
    print(f"Total number of data points: {len(reconstruction_errors)}")

    if len(anomaly_indices) == 0:
        print("No anomalies detected. Exiting.")
        return

    # After calculating reconstruction errors
    reconstructed_data = tcnatt_model(test_sequences.to(device))[0].cpu().detach().numpy()
    reconstructed_data = reconstructed_data.reshape(-1, reconstructed_data.shape[-1])

    gp_sequences = test_sequences.to('cpu')
    gp_errors = torch.tensor(reconstruction_errors, dtype=torch.float32).unsqueeze(1).to('cpu')

    gp_training_sequences = gp_sequences.reshape(len(gp_sequences), -1)  # Flatten the sequences
    gp_training_input = torch.cat([gp_training_sequences, gp_errors], dim=1)

    print(f"Shape of gp_sequences: {gp_sequences.shape}")
    print(f"Shape of gp_training_sequences after reshape: {gp_training_sequences.shape}")
    print(f"Shape of gp_errors: {gp_errors.shape}")
    print(f"Shape of gp_training_input: {gp_training_input.shape}")

    # Plot TCN-ATT model anomalies
    plot_tcnatt_anomalies(normalized_test_data, reconstructed_data, anomaly_indices, test_timestamps)

    # Step 5: Initialize and train GP model on all data
    num_inducing = min(500, gp_training_input.size(0))
    inducing_points = gp_training_input[:num_inducing]
    gp_model = VariationalGPModel(
        inducing_points,
        rank=1,  # Reduced rank for better stability
        min_size=128  # Increased min_size
    ).to(device)

    # Initialize likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    # Use a more stable optimizer
    optimizer = torch.optim.AdamW([
        {'params': gp_model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=learning_rate, weight_decay=weight_decay)

    # Train GP model
    gp_model.train()
    likelihood.train()
    mll = gpytorch.mlls.VariationalELBO(likelihood, gp_model, num_data=gp_training_input.size(0))

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = gp_model(gp_training_input)
        loss = -mll(output, gp_training_input[:, -1])  # Use last column as target (reconstruction errors)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(gp_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} - Loss: {loss.item()}')

    # Save the trained GP model
    save_path = 'trained_gp_model50.pth'
    torch.save({
        'gp_model_state_dict': gp_model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'inducing_points': inducing_points,
        'loss': loss.item(),
    }, save_path)
    print(f"Trained GP model saved to {os.path.abspath(save_path)}")

    # Step 6: Use GP to quantify uncertainty during inference with buffer processing
    gp_inference_inputs = prepare_gp_input_inference(sequences=gp_sequences, errors=gp_errors, anomaly_indices = anomaly_indices, gp_sequence_length=32, buffer_size=32, real_time_emulation=real_time_emulation, step_size=1)
    print(f"Number of GP inference buffers: {len(gp_inference_inputs)}")

    # Process GP inference inputs in batches
    uncertainties = []
    
    print("\nProcessing GP inference inputs:")
    for idx, gp_input in enumerate(gp_inference_inputs):
        try:
            # Ensure input is on the correct device
            gp_input = gp_input.to(device)
            
            # Process with GP model
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                output = gp_model(gp_input)
                variance = output.variance
                uncertainties.extend(variance.cpu().numpy().flatten())
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(gp_inference_inputs)} inputs")
                
        except Exception as e:
            print(f"Error processing input {idx}: {e}")
            continue
    
    # Convert uncertainties to numpy array
    uncertainties = np.array(uncertainties)
    
    print(f"\nFinal uncertainties shape: {uncertainties.shape}")
    
    # Create a full array of uncertainties
    full_uncertainties = np.zeros(len(normalized_test_data))
    
    # Handle the size mismatch by padding or truncating
    if len(uncertainties) < len(anomaly_indices):
        # Pad with the mean uncertainty if we have fewer values than anomalies
        padding_size = len(anomaly_indices) - len(uncertainties)
        mean_uncertainty = np.mean(uncertainties)
        uncertainties = np.pad(uncertainties, (0, padding_size), 
                             mode='constant', constant_values=mean_uncertainty)
    elif len(uncertainties) > len(anomaly_indices):
        # Truncate if we have more values than anomalies
        uncertainties = uncertainties[:len(anomaly_indices)]
    
    print(f"Adjusted uncertainties shape: {uncertainties.shape}")
    print(f"Number of anomaly indices: {len(anomaly_indices)}")
    
    # Now assign the uncertainties
    full_uncertainties[anomaly_indices] = uncertainties
    
    # Analyze uncertainties and assign traffic lights
    thresholds = analyze_uncertainties(full_uncertainties, gp_model, gp_training_input.to(device))
    traffic_lights = assign_traffic_lights(full_uncertainties, thresholds)

    # Step 9: Prepare results
    results = {
        'anomaly_indices': anomaly_indices,
        'uncertainties': uncertainties,
        'thresholds': thresholds,
        'traffic_lights': traffic_lights
    }

    # Print summary
    print(f"\nTotal data points: {len(normalized_test_data)}")
    print(f"Total anomalies detected: {len(anomaly_indices)} ({len(anomaly_indices)/len(normalized_test_data)*100:.2f}%)")
    print(f"Uncertainty levels: Min = {np.min(uncertainties):.6f}, Max = {np.max(uncertainties):.6f}, Avg = {np.mean(uncertainties):.6f}")
    print(f"Traffic light distribution: Red = {traffic_lights.count('red')}, Yellow = {traffic_lights.count('yellow')}, Green = {traffic_lights.count('green')}")

    # After training the GP model

    # After training the GP model
    output_scale = gp_model.covar_module.outputscale.item()
    #length_scales = gp_model.covar_module.base_kernel.lengthscale.cpu().numpy()

    print(f"GP Model Output Scale: {output_scale:.4f}")
    #print(f"GP Model Length Scales: {length_scales}")
    #print(f"Mean Length Scale: {length_scales.mean():.4f}")
    #print(f"Min Length Scale: {length_scales.min():.4f}")
    #print(f"Max Length Scale: {length_scales.max():.4f}")

    # Plot anomalies and predictions using Plotly with Traffic Lights
    plot_anomalies_and_predictions_plotly(normalized_test_data, reconstructed_data, anomaly_indices, traffic_lights, test_timestamps)
    plot_results_with_traffic_lights(normalized_test_data, test_timestamps, anomaly_indices, uncertainties, traffic_lights)

    # After processing GP inputs and before returning results
    print("\nCreating 3D visualization of GP distribution...")
    try:
        # Select a subset of data points if the dataset is too large
        max_points = 1000
        if len(gp_inference_inputs) > max_points:
            indices = np.random.choice(len(gp_inference_inputs), max_points, replace=False)
            vis_inputs = torch.cat([gp_inference_inputs[i] for i in indices], dim=0)
            vis_uncertainties = uncertainties[indices]
        else:
            vis_inputs = torch.cat(gp_inference_inputs, dim=0)
            vis_uncertainties = uncertainties

        plot_gp_3d_distribution(
            gp_model, 
            vis_inputs, 
            vis_uncertainties,
            title="GP Multivariate Distribution with Uncertainties"
        )
    except Exception as e:
        print(f"Warning: Could not create 3D visualization: {e}")

    return results

if __name__ == "__main__":
    test_data_path = r"C:\Users\Stell\Desktop\Traffic Lights UQ\Traffic Lights UQ\rig 2 run 4 envelope 12 features.csv"
    results = main(test_data_path)
















