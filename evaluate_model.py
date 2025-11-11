import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns

# Import necessary classes from train_best_model
from train_best_model import create_batch_edges

# Define AgriculturalDataset here (same as in train_best_model)
class AgriculturalDataset:
    """Process agricultural data for GAT-RNN training"""
    def __init__(self, df, encoders=None, scalers=None, training=True):
        self.df = df.copy()
        self.training = training
        
        self.categorical_cols = ['Region', 'District', 'Crop']
        
        self.price_cols = [
            'Seed_Price_Per_Kg',
            'Fertilizer_Price_Per_Kg', 
            'Herbicide_Price_Per_Litre',
            'Pesticide_Price_Per_Litre',
            'Labor_Cost_Per_Day'
        ]
        
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
        
        self.temporal_features = self.df[self.temporal_cols].values.astype(float)
        self.graph_features = self.df[self.graph_cols].values.astype(float)
        
        if training:
            self.scaler_temporal = StandardScaler()
            self.scaler_graph = StandardScaler()
            
            self.scalers_y = {}
            self.y_scaled = np.zeros_like(self.y)
            
            for i, price_col in enumerate(self.price_cols):
                self.scalers_y[price_col] = StandardScaler()
                self.y_scaled[:, i:i+1] = self.scalers_y[price_col].fit_transform(self.y[:, i:i+1])
            
            self.temporal_scaled = self.scaler_temporal.fit_transform(self.temporal_features)
            self.graph_scaled = self.scaler_graph.fit_transform(self.graph_features)
        else:
            self.scaler_temporal = scalers['temporal']
            self.scaler_graph = scalers['graph']
            self.scalers_y = scalers['y']
            
            self.y_scaled = np.zeros_like(self.y)
            for i, price_col in enumerate(self.price_cols):
                self.y_scaled[:, i:i+1] = self.scalers_y[price_col].transform(self.y[:, i:i+1])
            
            self.temporal_scaled = self.scaler_temporal.transform(self.temporal_features)
            self.graph_scaled = self.scaler_graph.transform(self.graph_features)

# Import the CURRENT model architecture
class GATRNNHybridModel_Current(torch.nn.Module):
    """Current GAT-RNN model (before multi-task enhancement)"""
    def __init__(self, temporal_input_size=4, graph_input_size=7, hidden_size=64):
        super(GATRNNHybridModel_Current, self).__init__()
        
        from torch_geometric.nn import GATConv
        
        self.temporal_encoder = torch.nn.LSTM(
            temporal_input_size, hidden_size, num_layers=3,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        
        self.gat1 = GATConv(graph_input_size, hidden_size, heads=8)
        self.gat2 = GATConv(hidden_size * 8, hidden_size * 2, heads=4)
        self.gat3 = GATConv(hidden_size * 8, hidden_size, heads=1)
        
        self.graph_norm1 = torch.nn.LayerNorm(hidden_size * 8)
        self.graph_norm2 = torch.nn.LayerNorm(hidden_size * 8)
        self.graph_norm3 = torch.nn.LayerNorm(hidden_size)
        
        self.graph_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
        
        fusion_input_size = hidden_size * 2 + hidden_size
        
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(fusion_input_size, hidden_size * 2),
            torch.nn.LayerNorm(hidden_size * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            
            torch.nn.Linear(hidden_size // 2, 5)
        )
    
    def forward(self, temporal_x, graph_x, edge_index=None):
        rnn_out, _ = self.temporal_encoder(temporal_x.unsqueeze(1))
        rnn_out = rnn_out.squeeze(1)
        
        graph_out = self.gat1(graph_x, edge_index)
        graph_out = self.graph_norm1(graph_out)
        graph_out = torch.relu(graph_out)
        
        graph_out = self.gat2(graph_out, edge_index)
        graph_out = self.graph_norm2(graph_out)
        graph_out = torch.relu(graph_out)
        
        graph_out = self.gat3(graph_out, edge_index)
        graph_out = self.graph_norm3(graph_out)
        
        graph_out = self.graph_proj(graph_out)
        
        combined = torch.cat([rnn_out, graph_out], dim=1)
        output = self.fusion(combined)
        
        return output

def load_trained_model():
    """Load the trained model"""
    checkpoint = torch.load('models/gat_rnn_best_model.pth', map_location='cpu', weights_only=False)
    
    model = GATRNNHybridModel_Current(
        temporal_input_size=4,
        graph_input_size=7,
        hidden_size=64
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

def evaluate_and_compare():
    """Compare actual vs predicted prices"""
    print("="*70)
    print("📊 ACTUAL vs PREDICTED PRICE COMPARISON")
    print("="*70)
    
    model, checkpoint = load_trained_model()
    device = torch.device('cpu')
    
    test_df = pd.read_csv('test_dataset_cleaned.csv')
    print(f"\n✅ Loaded {len(test_df)} test samples")
    
    test_dataset = AgriculturalDataset(
        test_df,
        encoders=checkpoint['encoders'],
        scalers={
            'temporal': checkpoint['scaler_temporal'],
            'graph': checkpoint['scaler_graph'],
            'y': checkpoint['scalers_y']  # Changed from scaler_y to scalers_y
        },
        training=False
    )
    
    print("\n🔮 Generating predictions...")
    temporal_tensor = torch.FloatTensor(test_dataset.temporal_scaled)
    graph_tensor = torch.FloatTensor(test_dataset.graph_scaled)
    
    batch_size = len(test_dataset.temporal_scaled)
    edge_index = create_batch_edges(batch_size, nodes_per_graph=1, device=device)
    
    with torch.no_grad():
        predictions_scaled = model(temporal_tensor, graph_tensor, edge_index)
    
    # Inverse transform using separate scalers
    predictions = np.zeros_like(predictions_scaled.numpy())
    for i, price_col in enumerate(test_dataset.price_cols):
        predictions[:, i:i+1] = checkpoint['scalers_y'][price_col].inverse_transform(
            predictions_scaled.numpy()[:, i:i+1]
        )
    
    actual = test_dataset.y
    
    # Price names
    price_names = ['Seed_Price_Per_Kg', 'Fertilizer_Price_Per_Kg', 
                   'Herbicide_Price_Per_Litre', 'Pesticide_Price_Per_Litre', 
                   'Labor_Cost_Per_Day']
    
    # Create comparison DataFrame
    print("\n" + "="*90)
    print("📈 SAMPLE PREDICTIONS (First 10 rows)")
    print("="*90)
    
    for i, price_name in enumerate(price_names):
        print(f"\n{price_name}:")
        print(f"{'Sample':<8} {'Actual':>12} {'Predicted':>12} {'Difference':>12} {'Error %':>10}")
        print("-"*60)
        
        for j in range(min(10, len(actual))):
            actual_price = actual[j, i]
            pred_price = predictions[j, i]
            diff = pred_price - actual_price
            error_pct = (abs(diff) / actual_price) * 100
            
            print(f"{j+1:<8} {actual_price:>12,.0f} {pred_price:>12,.0f} {diff:>+12,.0f} {error_pct:>9.2f}%")
    
    # Overall statistics
    print("\n" + "="*90)
    print("📊 OVERALL STATISTICS")
    print("="*90)
    print(f"{'Price Type':<20} {'MAE':>12} {'RMSE':>12} {'R²':>10} {'MAPE':>10}")
    print("-"*70)
    
    for i, price_name in enumerate(price_names):
        mae = mean_absolute_error(actual[:, i], predictions[:, i])
        rmse = np.sqrt(mean_squared_error(actual[:, i], predictions[:, i]))
        r2 = r2_score(actual[:, i], predictions[:, i])
        mape = np.mean(np.abs((actual[:, i] - predictions[:, i]) / actual[:, i])) * 100
        
        print(f"{price_name:<20} {mae:>12,.0f} {rmse:>12,.0f} {r2:>10.4f} {mape:>9.2f}%")
    
    # Overall
    overall_mae = mean_absolute_error(actual.flatten(), predictions.flatten())
    overall_rmse = np.sqrt(mean_squared_error(actual.flatten(), predictions.flatten()))
    overall_r2 = r2_score(actual.flatten(), predictions.flatten())
    
    print("-"*70)
    print(f"{'OVERALL':<20} {overall_mae:>12,.0f} {overall_rmse:>12,.0f} {overall_r2:>10.4f}")
    print("="*90)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Actual vs Predicted Prices - Test Set', fontsize=16, fontweight='bold')
    
    for i, (ax, price_name) in enumerate(zip(axes.flat, price_names)):
        ax.scatter(actual[:, i], predictions[:, i], alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(actual[:, i].min(), predictions[:, i].min())
        max_val = max(actual[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Price (UGX)', fontsize=10)
        ax.set_ylabel('Predicted Price (UGX)', fontsize=10)
        ax.set_title(f'{price_name}\nR² = {r2_score(actual[:, i], predictions[:, i]):.4f}', 
                     fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(price_names) < 6:
        fig.delaxes(axes.flat[5])
    
    plt.tight_layout()
    plt.savefig('models/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    print("✅ Saved visualization to: models/actual_vs_predicted.png")
    
    # Create error distribution plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Prediction Error Distribution - Test Set', fontsize=16, fontweight='bold')
    
    for i, (ax, price_name) in enumerate(zip(axes.flat, price_names)):
        errors = predictions[:, i] - actual[:, i]
        error_pct = (errors / actual[:, i]) * 100
        
        ax.hist(error_pct, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Prediction Error (%)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{price_name}\nMean Error: {np.mean(error_pct):.2f}%', 
                     fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    if len(price_names) < 6:
        fig.delaxes(axes.flat[5])
    
    plt.tight_layout()
    plt.savefig('models/error_distribution.png', dpi=150, bbox_inches='tight')
    print("✅ Saved error distribution to: models/error_distribution.png")
    
    # Export detailed comparison to CSV
    comparison_df = pd.DataFrame()
    for i, price_name in enumerate(price_names):
        comparison_df[f'{price_name}_Actual'] = actual[:, i]
        comparison_df[f'{price_name}_Predicted'] = predictions[:, i]
        comparison_df[f'{price_name}_Error'] = predictions[:, i] - actual[:, i]
        comparison_df[f'{price_name}_Error_Pct'] = ((predictions[:, i] - actual[:, i]) / actual[:, i]) * 100
    
    comparison_df.to_csv('models/detailed_comparison.csv', index=False)
    print("✅ Saved detailed comparison to: models/detailed_comparison.csv")
    
    print("\n✅ Evaluation complete!")
    print("\n📁 Generated files:")
    print("  - models/actual_vs_predicted.png")
    print("  - models/error_distribution.png")
    print("  - models/detailed_comparison.csv")
    
    return actual, predictions

if __name__ == '__main__':
    actual, predictions = evaluate_and_compare()
    
    print("\n🎯 Model is ready for deployment!")
    print("   Review the visualizations above to ensure quality.")
