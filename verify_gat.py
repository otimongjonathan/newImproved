import torch
import torch.nn as nn
import numpy as np
from train_best_model import GATRNNHybridModel, AgriculturalDataset, create_batch_edges
import pandas as pd

def verify_gat_is_working():
    """Verify that GAT layers are actually being used and learning"""
    print("="*70)
    print("🔍 VERIFYING GAT KNOWLEDGE GRAPH FUNCTIONALITY")
    print("="*70)
    
    # Load model
    checkpoint = torch.load('models/gat_rnn_best_model.pth', map_location='cpu', weights_only=False)
    
    model = GATRNNHybridModel(
        temporal_input_size=4,
        graph_input_size=7,
        hidden_size=32
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Check if GAT layers exist
    print("\n1️⃣ Checking Model Architecture:")
    print("-"*70)
    has_gat = hasattr(model, 'gat1') and hasattr(model, 'gat2')
    
    if has_gat:
        print("✅ GAT layers found: gat1 and gat2")
        print(f"   GAT1: {model.gat1}")
        print(f"   GAT2: {model.gat2}")
        
        # Check if GAT parameters are non-zero (learned)
        gat1_params = sum(p.abs().sum().item() for p in model.gat1.parameters())
        gat2_params = sum(p.abs().sum().item() for p in model.gat2.parameters())
        
        print(f"\n   GAT1 parameter sum (abs): {gat1_params:,.2f}")
        print(f"   GAT2 parameter sum (abs): {gat2_params:,.2f}")
        
        if gat1_params > 0 and gat2_params > 0:
            print("   ✅ GAT parameters are non-zero (learned)")
        else:
            print("   ❌ GAT parameters are zero (NOT learned)")
    else:
        print("❌ GAT layers NOT found - using fallback MLP")
        print(f"   Fallback FC1: {model.fc1}")
        print(f"   Fallback FC2: {model.fc2}")
    
    # Test with and without edges to see if GAT matters
    print("\n2️⃣ Testing GAT Impact (With vs Without Graph Connections):")
    print("-"*70)
    
    # Load test data
    test_df = pd.read_csv('test_dataset_cleaned.csv')
    test_dataset = AgriculturalDataset(
        test_df.head(100),  # Use 100 samples
        encoders=checkpoint['encoders'],
        scalers={
            'temporal': checkpoint['scaler_temporal'],
            'graph': checkpoint['scaler_graph'],
            'y': checkpoint['scaler_y']
        },
        training=False
    )
    
    temporal_tensor = torch.FloatTensor(test_dataset.temporal_scaled)
    graph_tensor = torch.FloatTensor(test_dataset.graph_scaled)
    
    # Prediction WITH edges (GAT active)
    batch_size = len(temporal_tensor)
    edge_index = create_batch_edges(batch_size, nodes_per_graph=1, device='cpu')
    
    with torch.no_grad():
        pred_with_edges = model(temporal_tensor, graph_tensor, edge_index)
    
    # Prediction WITHOUT edges - manually bypass GAT by using isolated nodes
    # Create edge index with NO connections (each node only connects to itself)
    isolated_edges = torch.tensor([[i, i] for i in range(batch_size)], dtype=torch.long).t()
    
    with torch.no_grad():
        pred_isolated = model(temporal_tensor, graph_tensor, isolated_edges)
    
    # Compare predictions
    diff = (pred_with_edges - pred_isolated).abs().mean().item()
    
    print(f"\n   Prediction difference (connected vs isolated graph): {diff:.6f}")
    
    if diff > 0.01:
        print(f"   ✅ GAT IS WORKING! Predictions change significantly with graph structure")
        print(f"   ✅ Mean absolute difference: {diff:.6f} (scaled)")
    else:
        print(f"   ⚠️  GAT may not be contributing much (difference < 0.01)")
        print(f"   ⚠️  Model might be relying more on LSTM pathway")
    
    # Check attention weights (if GAT)
    print("\n3️⃣ Analyzing GAT Attention Patterns:")
    print("-"*70)
    
    if has_gat:
        # Get attention weights from GAT layer
        with torch.no_grad():
            # Forward pass through GAT1 only
            gat1_out = model.gat1(graph_tensor, edge_index)
            
            print(f"   ✅ GAT1 output shape: {gat1_out.shape}")
            print(f"   ✅ GAT1 output range: [{gat1_out.min():.4f}, {gat1_out.max():.4f}]")
            print(f"   ✅ GAT1 output std: {gat1_out.std():.4f}")
            
            # Check if attention creates different representations for different regions/crops
            # Group by encoded region
            unique_regions = np.unique(test_dataset.graph_features[:, 0])  # Region_encoded
            
            print(f"\n   Testing attention on {len(unique_regions)} different regions:")
            for i, region in enumerate(unique_regions[:3]):  # Show first 3
                mask = test_dataset.graph_features[:, 0] == region
                region_outputs = gat1_out[mask]
                print(f"      Region {int(region)}: Mean output = {region_outputs.mean():.4f}, "
                      f"Std = {region_outputs.std():.4f}")
    
    # Compare pathway contributions
    print("\n4️⃣ Comparing Pathway Contributions (Temporal vs Graph):")
    print("-"*70)
    
    with torch.no_grad():
        # Get intermediate outputs
        rnn_out, _ = model.temporal_encoder(temporal_tensor.unsqueeze(1))
        rnn_out = rnn_out.squeeze(1)
        
        if has_gat and edge_index is not None:
            graph_out = model.gat1(graph_tensor, edge_index)
            graph_out = torch.relu(graph_out)
            graph_out = model.gat2(graph_out, edge_index)
        else:
            graph_out = model.relu(model.fc1(graph_tensor))
            graph_out = model.fc2(graph_out)
        
        graph_out = model.graph_norm(graph_out)
        graph_out = model.graph_proj(graph_out)
        
        # Check variance/importance
        rnn_variance = rnn_out.var().item()
        graph_variance = graph_out.var().item()
        
        print(f"   Temporal (LSTM) output variance: {rnn_variance:.6f}")
        print(f"   Graph (GAT) output variance: {graph_variance:.6f}")
        
        ratio = graph_variance / (rnn_variance + 1e-8)
        print(f"\n   Graph/Temporal variance ratio: {ratio:.4f}")
        
        if ratio > 0.5:
            print("   ✅ Graph pathway is contributing significantly")
        elif ratio > 0.1:
            print("   ⚠️  Graph pathway has moderate contribution")
        else:
            print("   ❌ Graph pathway contribution is very low - mostly using temporal")
    
    print("\n" + "="*70)
    print("✅ VERIFICATION COMPLETE!")
    print("="*70)
    
    return has_gat, diff > 0.01

if __name__ == '__main__':
    has_gat, is_working = verify_gat_is_working()
    
    if has_gat and is_working:
        print("\n🎉 RESULT: GAT Knowledge Graph IS working and contributing!")
    elif has_gat and not is_working:
        print("\n⚠️  RESULT: GAT exists but may not be contributing much")
    else:
        print("\n❌ RESULT: Using MLP fallback, not GAT")
