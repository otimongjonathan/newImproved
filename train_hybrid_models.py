import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset

# Make torch_geometric optional
try:
    from torch_geometric.nn import GATConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not available. Using simplified model.")

def batch_mean_pooling(x, batch):
    """Pooling that works for both temporal and graph features"""
    batch_size = batch.max().item() + 1
    output = torch.zeros(batch_size, x.size(-1), device=x.device)
    for i in range(batch_size):
        mask = (batch == i)
        output[i] = x[mask].mean(dim=0)
    return output

def create_batch_edges(batch_size, nodes_per_graph=4, device='cpu'):
    """Create and cache edge indices for batched graphs"""
    if not hasattr(create_batch_edges, 'cache'):
        create_batch_edges.cache = {}
    
    cache_key = (batch_size, nodes_per_graph, device)
    if cache_key in create_batch_edges.cache:
        return create_batch_edges.cache[cache_key]
    
    edge_indices = []
    for b in range(batch_size):
        offset = b * nodes_per_graph
        for i in range(nodes_per_graph):
            for j in range(nodes_per_graph):
                if i != j:
                    edge_indices.append([i + offset, j + offset])
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long, device=device).t()
    create_batch_edges.cache[cache_key] = edge_index
    return edge_index

class AgriculturalDataset(Dataset):
    def _create_batch_edges(self, num_nodes=4):
        """Create edges for a single graph"""
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def __init__(self, df, encoders=None, scalers=None, training=False):
        print(f"Initializing dataset with {len(df)} samples...")
        self.data = df.copy()
        
        self.data['Date'] = pd.to_datetime(self.data.iloc[:, 0])
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Year'] = self.data['Date'].dt.year - 2020
        
        self.categorical_cols = ['Region', 'District', 'Crop']
        self.price_cols = [4, 5, 6, 7, 8]
        
        if training:
            self.encoders = {col: LabelEncoder() for col in self.categorical_cols}
            self.scalers = {
                'temporal': StandardScaler(),
                'graph': StandardScaler(),
                'target': StandardScaler()
            }
            
            # Feature preprocessing
            for col in self.categorical_cols:
                self.data[col] = self.encoders[col].fit_transform(self.data[col].astype(str))
            
            temporal_data = np.column_stack([
                self.data.iloc[:, self.price_cols].values.astype(float),
                self.data[['Month']].values / 12,
                self.data[['Year']].values / 5
            ])
            self.scalers['temporal'].fit(temporal_data)
            
            graph_data = np.column_stack([
                self.data[self.categorical_cols].values.astype(float),
                self.data.iloc[:, self.price_cols].values.astype(float)
            ])
            self.scalers['graph'].fit(graph_data)
            
            self.scalers['target'].fit(self.data.iloc[:, self.price_cols].values.astype(float))
        else:
            self.encoders = encoders
            self.scalers = scalers
            for col in self.categorical_cols:
                self.data[col] = self.encoders[col].transform(self.data[col].astype(str))
        
        self.batch_edges = self._create_batch_edges()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        temporal_features = np.concatenate([
            row.iloc[self.price_cols].values.astype(float),
            [row['Month'], row['Year']]
        ])
        temporal_features = self.scalers['temporal'].transform(
            temporal_features.reshape(1, -1)
        ).squeeze()
        
        graph_features = np.concatenate([
            [row[col] for col in self.categorical_cols],
            row.iloc[self.price_cols].values.astype(float)
        ])
        graph_features = self.scalers['graph'].transform(
            graph_features.reshape(1, -1)
        ).squeeze()
        
        return {
            'temporal': torch.FloatTensor(temporal_features),
            'graph': torch.FloatTensor(graph_features).repeat(4, 1),  # 4 nodes per graph
            'edge_index': self.batch_edges,
            'target': torch.FloatTensor(
                self.scalers['target'].transform(
                    row.iloc[self.price_cols].values.astype(float).reshape(1, -1)
                )
            ).squeeze()
        }

class GATRNNHybrid(nn.Module):
    def __init__(self, temporal_input_size, graph_input_size, hidden_size):
        super(GATRNNHybrid, self).__init__()
        
        # Temporal pathway - match checkpoint naming: temporal_encoder (not rnn)
        self.temporal_encoder = nn.LSTM(
            temporal_input_size, 
            hidden_size, 
            num_layers=2,  # checkpoint has 2 layers
            batch_first=True
        )
        
        # Graph pathway
        if TORCH_GEOMETRIC_AVAILABLE:
            self.gat1 = GATConv(graph_input_size, hidden_size, heads=4)
            self.gat2 = GATConv(hidden_size * 4, hidden_size, heads=1)
        else:
            self.fc1 = nn.Linear(graph_input_size, hidden_size * 4)
            self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
            self.relu = nn.ReLU()
        
        # Graph normalization
        self.graph_norm = nn.BatchNorm1d(hidden_size)
        
        # Graph projection layer (from checkpoint)
        self.graph_proj = nn.Linear(hidden_size, hidden_size)
        
        # Fusion layers - match checkpoint structure
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # fusion.0
            nn.BatchNorm1d(hidden_size),              # fusion.1
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 5)                 # fusion.4 - output layer
        )
    
    def forward(self, temporal_x, graph_x, edge_index=None, batch=None):
        # Temporal pathway
        rnn_out, _ = self.temporal_encoder(temporal_x.unsqueeze(1))
        rnn_out = rnn_out.squeeze(1)
        
        # Graph pathway
        if TORCH_GEOMETRIC_AVAILABLE and edge_index is not None:
            graph_out = self.gat1(graph_x, edge_index)
            graph_out = torch.relu(graph_out)
            graph_out = self.gat2(graph_out, edge_index)
        else:
            graph_out = self.relu(self.fc1(graph_x))
            graph_out = self.fc2(graph_out)
        
        # Normalize and project graph features
        graph_out = self.graph_norm(graph_out)
        graph_out = self.graph_proj(graph_out)
        
        # Combine temporal and graph features
        combined = torch.cat([rnn_out, graph_out], dim=1)
        
        # Fusion and output
        output = self.fusion(combined)
        
        return output

def evaluate_model(model, loader, criterion, device):
    """Evaluate model with detailed metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            temporal_x = batch['temporal'].to(device)
            graph_x = batch['graph'].to(device)
            targets = batch['target'].to(device)
            batch_size = temporal_x.size(0)
            
            edge_index = create_batch_edges(batch_size).to(device)
            batch_idx = torch.arange(batch_size, device=device).repeat_interleave(4)
            
            outputs = model(temporal_x, graph_x, edge_index, batch_idx)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Combine predictions and targets
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / len(loader),
        'r2': r2_score(targets, predictions),
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'mae': mean_absolute_error(targets, predictions),
        'per_output_r2': [
            r2_score(targets[:, i], predictions[:, i]) 
            for i in range(targets.shape[1])
        ]
    }
    
    return metrics

def train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, 
                      criterion, device, num_epochs=50):  # Increased epochs
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    model.train()
    best_val_loss = float('inf')
    
    history = {
        'train_loss': [], 
        'val_loss': [],
        'val_r2': [],
        'learning_rates': []
    }
    
    for epoch in tqdm(range(num_epochs)):
        # Training loop
        model.train()
        train_loss = 0
        for batch in train_loader:
            temporal_x = batch['temporal'].to(device)
            graph_x = batch['graph'].to(device)
            targets = batch['target'].to(device)  # Added targets here
            batch_size = temporal_x.size(0)
            
            # Create proper edge indices for the batch
            edge_index = create_batch_edges(batch_size).to(device)
            batch_idx = torch.arange(batch_size, device=device).repeat_interleave(4)
            
            optimizer.zero_grad()
            outputs = model(temporal_x, graph_x, edge_index, batch_idx)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Validate every 3 epochs
        if epoch % 3 == 0:
            val_metrics = evaluate_model(model, val_loader, criterion, device)
            avg_val_loss = val_metrics['loss']
            
            print(f"\nEpoch {epoch + 1}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"Val R²: {val_metrics['r2']:.4f}")
            print(f"Val RMSE: {val_metrics['rmse']:.4f}")
            
            history['val_loss'].append(avg_val_loss)
            history['val_r2'].append(val_metrics['r2'])
            
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save only model weights
                torch.save(model.state_dict(), 'models/best_model.pth')
                # Save metadata separately
                metadata = {
                    'epoch': epoch,
                    'val_metrics': val_metrics,
                    'history': history
                }
                with open('models/best_model_meta.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
    
    # Final evaluation
    print("\nFinal Model Evaluation:")
    print("-" * 40)
    
    # Load best model weights
    model.load_state_dict(torch.load('models/best_model.pth'))
    
    # Load metadata if needed
    try:
        with open('models/best_model_meta.json', 'r') as f:
            metadata = json.load(f)
        print(f"Best model from epoch: {metadata['epoch'] + 1}")
    except:
        print("Metadata not found, continuing with evaluation...")
    
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    print(f"Test R²: {test_metrics['r2']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.4f}")
    print("\nPer-output R² scores:")
    for i, r2 in enumerate(test_metrics['per_output_r2']):
        print(f"Output {i+1}: {r2:.4f}")
    
    # Save final results
    results = {
        'test_metrics': test_metrics,
        'training_history': history
    }
    
    with open('models/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to models/training_results.json")
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Smaller batch size for CPU
    batch_size = 32 if device.type == 'cpu' else 64
    
    print("Loading datasets...")
    with tqdm(total=3, desc="Loading data") as pbar:
        train_df = pd.read_csv('train_dataset_cleaned.csv')
        pbar.update(1)
        val_df = pd.read_csv('validation_dataset_cleaned.csv')
        pbar.update(1)
        test_df = pd.read_csv('test_dataset_cleaned.csv')
        pbar.update(1)
    
    # Create datasets with progress tracking
    train_dataset = AgriculturalDataset(train_df, training=True)
    val_dataset = AgriculturalDataset(val_df, 
                                    encoders=train_dataset.encoders,
                                    scalers=train_dataset.scalers)
    test_dataset = AgriculturalDataset(test_df,
                                     encoders=train_dataset.encoders,
                                     scalers=train_dataset.scalers)
    
    # Optimize data loading for CPU
    num_workers = 0 if device.type == 'cpu' else 4
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False  # Disabled for CPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    # Cache edge indices for common batch sizes
    print("Pre-computing edge indices...")
    for bs in [batch_size, batch_size * 2]:
        _ = create_batch_edges(bs, device=device)
    
    model = GATRNNHybrid(
        temporal_input_size=7,
        graph_input_size=8,
        hidden_size=8  # Further reduced hidden size
    ).to(device)
    
    # Faster optimizer settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.005,  # Increased learning rate
        weight_decay=0.001
    )
    criterion = nn.MSELoss()
    
    os.makedirs('models', exist_ok=True)
    
    train_and_evaluate(
        model, train_loader, val_loader, test_loader,
        optimizer, criterion, device
    )

if __name__ == '__main__':
    main()
