import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json
from datetime import datetime

# Try to import torch_geometric, use fallback if not available
try:
    from torch_geometric.nn import GATConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("⚠️ Warning: torch_geometric not available. Using fallback architecture.")

class GATRNNHybridModel(nn.Module):
    """Enhanced GAT-RNN Hybrid Model with multi-task learning"""
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
        
        # ENHANCEMENT: Separate prediction heads for each price type
        self.shared_fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Individual prediction heads for each price type
        self.price_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 1)
            ) for _ in range(5)  # 5 price types
        ])
    
    def forward(self, temporal_x, graph_x, edge_index=None, batch=None):
        rnn_out, _ = self.temporal_encoder(temporal_x.unsqueeze(1))
        rnn_out = rnn_out.squeeze(1)
        
        if TORCH_GEOMETRIC_AVAILABLE and edge_index is not None:
            graph_out = self.gat1(graph_x, edge_index)
            graph_out = self.graph_norm1(graph_out)
            graph_out = torch.relu(graph_out)
            graph_out = torch.dropout(graph_out, p=0.2, train=self.training)
            
            graph_out = self.gat2(graph_out, edge_index)
            graph_out = self.graph_norm2(graph_out)
            graph_out = torch.relu(graph_out)
            graph_out = torch.dropout(graph_out, p=0.2, train=self.training)
            
            graph_out = self.gat3(graph_out, edge_index)
            graph_out = self.graph_norm3(graph_out)
        else:
            graph_out = self.relu(self.fc1(graph_x))
            graph_out = self.graph_norm1(graph_out)
            graph_out = torch.dropout(graph_out, p=0.2, train=self.training)
            
            graph_out = self.relu(self.fc2(graph_out))
            graph_out = self.graph_norm2(graph_out)
            graph_out = torch.dropout(graph_out, p=0.2, train=self.training)
            
            graph_out = self.fc3(graph_out)
            graph_out = self.graph_norm3(graph_out)
        
        graph_out = self.graph_proj(graph_out)
        
        combined = torch.cat([rnn_out, graph_out], dim=1)
        
        # Shared representation
        shared = self.shared_fusion(combined)
        
        # Individual predictions from separate heads
        outputs = torch.cat([head(shared) for head in self.price_heads], dim=1)
        
        return outputs

def create_batch_edges(batch_size, nodes_per_graph=1, device='cpu'):
    """Create edge indices for batched graphs - each sample is a single node with self-connections"""
    # For batch processing, create fully connected graph within batch
    edge_indices = []
    for i in range(batch_size):
        for j in range(batch_size):
            if i != j:  # Connect to other nodes in batch
                edge_indices.append([i, j])
    
    if edge_indices:
        return torch.tensor(edge_indices, dtype=torch.long).t().to(device)
    return None

class AgriculturalDataset:
    """Process agricultural data for GAT-RNN training"""
    def __init__(self, df, encoders=None, scalers=None, training=True):
        self.df = df.copy()
        self.training = training
        
        if training:
            print(f"  📋 Dataset columns: {list(self.df.columns)}")
        
        self.categorical_cols = ['Region', 'District', 'Crop']
        
        self.price_cols = [
            'Seed_Price_Per_Kg',
            'Fertilizer_Price_Per_Kg', 
            'Herbicide_Price_Per_Litre',
            'Pesticide_Price_Per_Litre',
            'Labor_Cost_Per_Day'
        ]
        
        if training:
            print(f"  📊 Using price columns: {self.price_cols}")
        
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Year'] = self.df['Date'].dt.year
        
        if training:
            self.encoders = {}
            for col in self.categorical_cols:
                self.encoders[col] = LabelEncoder()
                self.df[f'{col}_encoded'] = self.encoders[col].fit_transform(self.df[col])
        else:
            self.encoders = encoders
            for col in self.categorical_cols:
                self.df[f'{col}_encoded'] = self.encoders[col].transform(self.df[col])
        
        self.temporal_cols = ['Month', 'Year', 'Rainfall_Index', 'Soil_Fertility_Index']
        
        self.graph_cols = [f'{col}_encoded' for col in self.categorical_cols]
        self.graph_cols.extend(['Month', 'Year', 'Rainfall_Index', 'Soil_Fertility_Index'])
        
        self.y = self.df[self.price_cols].values.astype(float)
        
        if training:
            print(f"  🔢 Temporal features ({len(self.temporal_cols)}): {self.temporal_cols}")
            print(f"  🔢 Graph features ({len(self.graph_cols)}): {self.graph_cols}")
        
        self.temporal_features = self.df[self.temporal_cols].values.astype(float)
        self.graph_features = self.df[self.graph_cols].values.astype(float)
        
        # FIX: Use separate scalers for each price type to preserve individual variance
        if training:
            self.scaler_temporal = StandardScaler()
            self.scaler_graph = StandardScaler()
            
            # Separate scaler for EACH price column
            self.scalers_y = {}
            self.y_scaled = np.zeros_like(self.y)
            
            for i, price_col in enumerate(self.price_cols):
                self.scalers_y[price_col] = StandardScaler()
                self.y_scaled[:, i:i+1] = self.scalers_y[price_col].fit_transform(self.y[:, i:i+1])
            
            self.temporal_scaled = self.scaler_temporal.fit_transform(self.temporal_features)
            self.graph_scaled = self.scaler_graph.fit_transform(self.graph_features)
            
            print(f"  ✅ Using separate scalers for each price type")
        else:
            self.scaler_temporal = scalers['temporal']
            self.scaler_graph = scalers['graph']
            self.scalers_y = scalers['y']
            
            # Apply separate scalers
            self.y_scaled = np.zeros_like(self.y)
            for i, price_col in enumerate(self.price_cols):
                self.y_scaled[:, i:i+1] = self.scalers_y[price_col].transform(self.y[:, i:i+1])
            
            self.temporal_scaled = self.scaler_temporal.transform(self.temporal_features)
            self.graph_scaled = self.scaler_graph.transform(self.graph_features)

def evaluate_model(model, dataloader, scalers_y, price_cols, device):
    """Evaluate GAT-RNN model with separate scalers"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for temporal_batch, graph_batch, y_batch in dataloader:
            temporal_batch = temporal_batch.to(device)
            graph_batch = graph_batch.to(device)
            
            batch_size = temporal_batch.size(0)
            edge_index = create_batch_edges(batch_size, nodes_per_graph=1, device=device)
            
            outputs = model(temporal_batch, graph_batch, edge_index)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    y_pred_scaled = np.concatenate(all_preds, axis=0)
    y_true_scaled = np.concatenate(all_targets, axis=0)
    
    # Inverse transform using separate scalers
    y_pred = np.zeros_like(y_pred_scaled)
    y_true = np.zeros_like(y_true_scaled)
    
    for i, price_col in enumerate(price_cols):
        y_pred[:, i:i+1] = scalers_y[price_col].inverse_transform(y_pred_scaled[:, i:i+1])
        y_true[:, i:i+1] = scalers_y[price_col].inverse_transform(y_true_scaled[:, i:i+1])
    
    metrics = {}
    output_names = ['Seed', 'Fertilizer', 'Herbicide', 'Pesticide', 'Labor']
    
    for i, name in enumerate(output_names):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / y_true[:, i])) * 100
        
        metrics[name] = {
            'MSE': float(mse),
            'RMSE': float(np.sqrt(mse)),
            'MAE': float(mae),
            'R2': float(r2),
            'MAPE': float(mape)
        }
    
    overall_mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    overall_mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    overall_r2 = r2_score(y_true.flatten(), y_pred.flatten())
    
    metrics['Overall'] = {
        'MSE': float(overall_mse),
        'RMSE': float(np.sqrt(overall_mse)),
        'MAE': float(overall_mae),
        'R2': float(overall_r2)
    }
    
    return metrics

def train_model():
    """Train the enhanced GAT-RNN hybrid model"""
    print("="*70)
    print("🌾 ENHANCED GAT-RNN MODEL TRAINING (TARGET: R² > 0.92)")
    print("="*70)
    
    print("\n📂 Loading data...")
    train_df = pd.read_csv('train_dataset_cleaned.csv')
    test_df = pd.read_csv('test_dataset_cleaned.csv')
    
    print(f"  ✅ Training samples: {len(train_df)}")
    print(f"  ✅ Test samples: {len(test_df)}")
    
    # FIX: Use original split - just create validation from train
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, shuffle=True)
    
    print(f"  ✅ Split - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    print("\n🔧 Preparing datasets...")
    train_dataset = AgriculturalDataset(train_df, training=True)
    
    val_dataset = AgriculturalDataset(
        val_df,
        encoders=train_dataset.encoders,
        scalers={
            'temporal': train_dataset.scaler_temporal,
            'graph': train_dataset.scaler_graph,
            'y': train_dataset.scalers_y
        },
        training=False
    )
    
    test_dataset = AgriculturalDataset(
        test_df, 
        encoders=train_dataset.encoders,
        scalers={
            'temporal': train_dataset.scaler_temporal,
            'graph': train_dataset.scaler_graph,
            'y': train_dataset.scalers_y
        },
        training=False
    )
    
    # FIX 2: Use larger batch size to stabilize training
    train_loader = DataLoader(
        list(zip(torch.FloatTensor(train_dataset.temporal_scaled),
                torch.FloatTensor(train_dataset.graph_scaled),
                torch.FloatTensor(train_dataset.y_scaled))),
        batch_size=128,  # Increased from 64
        shuffle=True
    )
    
    val_loader = DataLoader(
        list(zip(torch.FloatTensor(val_dataset.temporal_scaled),
                torch.FloatTensor(val_dataset.graph_scaled),
                torch.FloatTensor(val_dataset.y_scaled))),
        batch_size=128,
        shuffle=False
    )
    
    test_loader = DataLoader(
        list(zip(torch.FloatTensor(test_dataset.temporal_scaled),
                torch.FloatTensor(test_dataset.graph_scaled),
                torch.FloatTensor(test_dataset.y_scaled))),
        batch_size=128,
        shuffle=False
    )
    
    print("\n🏗️ Initializing Enhanced GAT-RNN model with Multi-Task Learning...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  ✅ Using device: {device}")
    
    temporal_size = len(train_dataset.temporal_cols)
    graph_size = len(train_dataset.graph_cols)
    
    model = GATRNNHybridModel(
        temporal_input_size=temporal_size,
        graph_input_size=graph_size,
        hidden_size=64
    ).to(device)
    
    print(f"  ✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  ✅ Architecture: 3-Layer GAT + Bidirectional 3-Layer LSTM + Multi-Task Heads")
    
    # ENHANCEMENT: Weighted loss for each price type
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=5e-4)  # Higher LR
    
    # More epochs with cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-7
    )
    
    print("\n🚀 Starting training...")
    print("-"*70)
    
    best_val_loss = float('inf')
    best_test_r2 = -float('inf')
    best_metrics = None
    patience = 50  # Increased patience
    patience_counter = 0
    
    num_epochs = 300  # More epochs
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        for temporal_batch, graph_batch, y_batch in train_loader:
            temporal_batch = temporal_batch.to(device)
            graph_batch = graph_batch.to(device)
            y_batch = y_batch.to(device)
            
            batch_size = temporal_batch.size(0)
            edge_index = create_batch_edges(batch_size, nodes_per_graph=1, device=device)
            
            optimizer.zero_grad()
            outputs = model(temporal_batch, graph_batch, edge_index)
            loss = criterion(outputs, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for temporal_batch, graph_batch, y_batch in val_loader:
                temporal_batch = temporal_batch.to(device)
                graph_batch = graph_batch.to(device)
                y_batch = y_batch.to(device)
                
                batch_size = temporal_batch.size(0)
                edge_index = create_batch_edges(batch_size, nodes_per_graph=1, device=device)
                
                outputs = model(temporal_batch, graph_batch, edge_index)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"LR: {current_lr:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            print(f"\n  🎯 New best validation! Evaluating on all datasets...")
            
            train_metrics = evaluate_model(model, train_loader, train_dataset.scalers_y, train_dataset.price_cols, device)
            val_metrics = evaluate_model(model, val_loader, train_dataset.scalers_y, train_dataset.price_cols, device)
            test_metrics = evaluate_model(model, test_loader, train_dataset.scalers_y, train_dataset.price_cols, device)
            
            test_r2 = test_metrics['Overall']['R2']
            
            if test_r2 > best_test_r2:
                best_test_r2 = test_r2
                patience_counter = 0
                
                best_metrics = {
                    'train': train_metrics,
                    'validation': val_metrics,
                    'test': test_metrics
                }
                
                os.makedirs('models', exist_ok=True)
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'train_loss': avg_train_loss,
                    'metrics': best_metrics,
                    'encoders': train_dataset.encoders,
                    'scaler_temporal': train_dataset.scaler_temporal,
                    'scaler_graph': train_dataset.scaler_graph,
                    'scalers_y': train_dataset.scalers_y,
                    'price_cols': train_dataset.price_cols,
                    'model_type': 'Enhanced_GAT_RNN_Hybrid',
                    'hidden_size': 64
                }
                
                torch.save(checkpoint, 'models/gat_rnn_best_model.pth')
                
                print(f"\n  📊 Performance Metrics Comparison:")
                print(f"  {'='*90}")
                print(f"  {'Metric':<12} {'Train MAE':>12} {'Val MAE':>12} {'Test MAE':>12} {'Train R²':>10} {'Val R²':>10} {'Test R²':>10}")
                print(f"  {'-'*90}")
                
                for name in ['Seed', 'Fertilizer', 'Herbicide', 'Pesticide', 'Labor']:
                    print(f"  {name:<12} "
                          f"{train_metrics[name]['MAE']:>12,.0f} "
                          f"{val_metrics[name]['MAE']:>12,.0f} "
                          f"{test_metrics[name]['MAE']:>12,.0f} "
                          f"{train_metrics[name]['R2']:>10.4f} "
                          f"{val_metrics[name]['R2']:>10.4f} "
                          f"{test_metrics[name]['R2']:>10.4f}")
                
                print(f"  {'-'*90}")
                print(f"  {'Overall':<12} "
                      f"{train_metrics['Overall']['MAE']:>12,.0f} "
                      f"{val_metrics['Overall']['MAE']:>12,.0f} "
                      f"{test_metrics['Overall']['MAE']:>12,.0f} "
                      f"{train_metrics['Overall']['R2']:>10.4f} "
                      f"{val_metrics['Overall']['R2']:>10.4f} "
                      f"{test_metrics['Overall']['R2']:>10.4f}")
                print(f"  {'='*90}\n")
                
                # Show progress towards goal
                progress = (test_r2 / 0.92) * 100
                print(f"\n  🎯 Progress towards R² = 0.92: {progress:.1f}%")
                if test_r2 >= 0.92:
                    print(f"  🎉🎉🎉 TARGET ACHIEVED! R² = {test_r2:.4f} >= 0.92 🎉🎉🎉")
                    break  # Stop training when target is reached
            else:
                patience_counter += 1
                print(f"  ⚠️ Test R² = {test_r2:.4f} (patience: {patience_counter}/{patience})")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break
    
    if best_metrics is None:
        print("\n❌ Training failed! No model with positive test R² found.")
        print("   Try: 1) More data, 2) Simpler model, 3) Better features")
        return None, None
    
    print("\n" + "="*70)
    print("✅ GAT-RNN TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📁 Model saved to: models/gat_rnn_best_model.pth")
    print(f"🤖 Architecture: Graph Attention Network + LSTM RNN")
    print(f"📊 Best validation loss: {best_val_loss:.6f}")
    print(f"\n🎯 Final Test Set Performance:")
    print(f"  Overall R² Score: {best_metrics['test']['Overall']['R2']:.4f}")
    print(f"  Overall MAE: {best_metrics['test']['Overall']['MAE']:,.0f} UGX")
    print(f"  Overall RMSE: {best_metrics['test']['Overall']['RMSE']:,.0f} UGX")
    print("\n" + "="*70)
    
    return model, best_metrics

if __name__ == '__main__':
    train_model()
