import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

try:
    from torch_geometric.nn import GATConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

class GATRNNHybridModel(nn.Module):
    """Enhanced GAT-RNN Hybrid Model - matches trained model"""
    def __init__(self, temporal_input_size=4, graph_input_size=7, hidden_size=64):
        super(GATRNNHybridModel, self).__init__()
        
        self.temporal_encoder = nn.LSTM(
            temporal_input_size, 
            hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        if TORCH_GEOMETRIC_AVAILABLE:
            self.gat1 = GATConv(graph_input_size, hidden_size, heads=8)
            self.gat2 = GATConv(hidden_size * 8, hidden_size * 2, heads=4)
            self.gat3 = GATConv(hidden_size * 8, hidden_size, heads=1)
        else:
            self.fc1 = nn.Linear(graph_input_size, hidden_size * 8)
            self.fc2 = nn.Linear(hidden_size * 8, hidden_size * 8)
            self.fc3 = nn.Linear(hidden_size * 8, hidden_size)
        
        # Add relu as attribute for fallback path
        self.relu = nn.ReLU()
        
        self.graph_norm1 = nn.LayerNorm(hidden_size * 8)
        self.graph_norm2 = nn.LayerNorm(hidden_size * 8)
        self.graph_norm3 = nn.LayerNorm(hidden_size)
        
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        fusion_input_size = hidden_size * 2 + hidden_size
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_size // 2, 5)
        )
    
    def forward(self, temporal_x, graph_x, edge_index=None):
        # Bidirectional LSTM
        rnn_out, _ = self.temporal_encoder(temporal_x.unsqueeze(1))
        rnn_out = rnn_out.squeeze(1)
        
        if TORCH_GEOMETRIC_AVAILABLE and edge_index is not None:
            graph_out = self.gat1(graph_x, edge_index)
            graph_out = self.graph_norm1(graph_out)
            graph_out = torch.relu(graph_out)
            
            graph_out = self.gat2(graph_out, edge_index)
            graph_out = self.graph_norm2(graph_out)
            graph_out = torch.relu(graph_out)
            
            graph_out = self.gat3(graph_out, edge_index)
            graph_out = self.graph_norm3(graph_out)
        else:
            graph_out = self.relu(self.fc1(graph_x))
            graph_out = self.graph_norm1(graph_out)
            
            graph_out = self.relu(self.fc2(graph_out))
            graph_out = self.graph_norm2(graph_out)
            
            graph_out = self.fc3(graph_out)
            graph_out = self.graph_norm3(graph_out)
        
        graph_out = self.graph_proj(graph_out)
        
        combined = torch.cat([rnn_out, graph_out], dim=1)
        output = self.fusion(combined)
        
        return output

def load_model_and_dataset():
    """Load the trained GAT-RNN model"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'models', 'gat_rnn_best_model.pth')
    
    print(f"📂 Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Initialize model with correct architecture
    model = GATRNNHybridModel(
        temporal_input_size=4,
        graph_input_size=7,
        hidden_size=checkpoint.get('hidden_size', 64)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Model loaded successfully!")
    print(f"🎯 Test R² Score: {checkpoint['metrics']['test']['Overall']['R2']:.4f}")
    print(f"📊 Test MAE: {checkpoint['metrics']['test']['Overall']['MAE']:.0f} UGX")
    
    # Get encoders and scalers (now uses scalers_y dict, not scaler_y)
    encoders = checkpoint['encoders']
    scaler_temporal = checkpoint['scaler_temporal']
    scaler_graph = checkpoint['scaler_graph']
    scalers_y = checkpoint['scalers_y']  # Changed from scaler_y
    price_cols = checkpoint['price_cols']
    
    return model, encoders, scaler_temporal, scaler_graph, scalers_y, price_cols, device

def predict_future_prices(model, encoders, scaler_temporal, scaler_graph, scalers_y, price_cols, device,
                          region, district, crop, acres, forecast_months):
    """Make price predictions for future months"""
    
    predictions = []
    current_date = datetime.now()
    
    # Encode inputs
    region_enc = encoders['Region'].transform([region])[0]
    district_enc = encoders['District'].transform([district])[0]
    crop_enc = encoders['Crop'].transform([crop])[0]
    
    # Default environmental values
    rainfall_index = 0.6
    soil_fertility = 0.7
    
    for i in range(forecast_months):
        future_date = current_date + relativedelta(months=i)
        month = future_date.month
        year = future_date.year
        
        # Temporal features: [Month, Year, Rainfall, Soil_Fertility]
        temporal_features = np.array([[month, year, rainfall_index, soil_fertility]])
        
        # Graph features: [Region_enc, District_enc, Crop_enc, Month, Year, Rainfall, Soil_Fertility]
        graph_features = np.array([[region_enc, district_enc, crop_enc, month, year, rainfall_index, soil_fertility]])
        
        # Scale features
        temporal_scaled = scaler_temporal.transform(temporal_features)
        graph_scaled = scaler_graph.transform(graph_features)
        
        # Convert to tensors
        temporal_tensor = torch.FloatTensor(temporal_scaled).to(device)
        graph_tensor = torch.FloatTensor(graph_scaled).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(temporal_tensor, graph_tensor, edge_index=None)
            
            # Inverse transform using separate scalers for each price
            prices = np.zeros(5)
            for idx, price_col in enumerate(price_cols):
                prices[idx] = scalers_y[price_col].inverse_transform(
                    output.cpu().numpy()[:, idx:idx+1]
                )[0][0]
        
        # Scale by acres
        predictions.append({
            'month': future_date.strftime('%B %Y'),
            'seed': float(prices[0]) * acres,
            'fertilizer': float(prices[1]) * acres,
            'herbicide': float(prices[2]) * acres,
            'pesticide': float(prices[3]) * acres,
            'labor': float(prices[4]) * acres
        })
    
    return predictions
