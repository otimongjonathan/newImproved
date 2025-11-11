import torch
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from train_hybrid_models import GATRNNHybrid, create_batch_edges, AgriculturalDataset
from torch.utils.data import DataLoader
import os
import gc
import requests
import gzip


INFLATION_RATE = 0.008
SUPPLY_CHAIN_VOLATILITY = 0.03


REGIONAL_INFLATION = {
    'Central': 1.0,
    'Eastern': 0.95,
    'Western': 1.05,
    'Northern': 1.1
}

WEATHER_RISK = {
    1: 1.05, 2: 1.08, 3: 1.12,
    4: 1.15, 5: 1.10, 6: 0.95,
    7: 0.90, 8: 0.92, 9: 0.95,
    10: 1.00, 11: 1.05, 12: 1.08
}

SEASONAL_MULTIPLIERS = {
    'Central': {
        'Maize': {
            'seed': [1.0, 1.2, 1.4, 1.3, 1.1, 0.8, 0.7, 0.9, 1.2, 1.1, 0.9, 0.8],
            'fertilizer': [1.0, 1.3, 1.5, 1.2, 1.0, 0.8, 0.7, 0.9, 1.3, 1.2, 0.9, 0.8],
            'herbicide': [0.8, 1.1, 1.4, 1.5, 1.2, 0.9, 0.7, 0.8, 1.1, 1.3, 1.0, 0.8],
            'pesticide': [0.8, 1.0, 1.3, 1.4, 1.2, 1.0, 0.8, 0.9, 1.2, 1.3, 1.0, 0.9],
            'labor': [1.0, 1.2, 1.4, 1.3, 1.1, 0.9, 0.8, 0.9, 1.2, 1.1, 1.0, 0.9]
        },
        'Coffee': {
            'seed': [0.9, 1.0, 1.2, 1.3, 1.1, 0.9, 0.8, 0.9, 1.0, 1.1, 1.0, 0.9],
            'fertilizer': [1.1, 1.3, 1.4, 1.2, 1.0, 0.9, 0.8, 1.0, 1.2, 1.3, 1.1, 1.0],
            'herbicide': [0.9, 1.1, 1.3, 1.4, 1.1, 0.9, 0.8, 0.9, 1.1, 1.2, 1.0, 0.9],
            'pesticide': [1.0, 1.2, 1.4, 1.3, 1.1, 0.9, 0.8, 1.0, 1.2, 1.3, 1.1, 1.0],
            'labor': [1.0, 1.1, 1.3, 1.4, 1.2, 1.0, 0.9, 1.0, 1.2, 1.3, 1.1, 1.0]
        },
        'Sweet Potatoes': {
            'seed': [1.0, 1.1, 1.3, 1.2, 1.0, 0.8, 0.7, 0.9, 1.1, 1.2, 1.0, 0.9],
            'fertilizer': [0.9, 1.1, 1.3, 1.2, 1.0, 0.8, 0.7, 0.9, 1.2, 1.1, 0.9, 0.8],
            'herbicide': [0.8, 1.0, 1.2, 1.3, 1.1, 0.9, 0.7, 0.8, 1.0, 1.2, 1.0, 0.9],
            'pesticide': [0.9, 1.0, 1.2, 1.3, 1.1, 0.9, 0.8, 0.9, 1.1, 1.2, 1.0, 0.9],
            'labor': [1.0, 1.1, 1.3, 1.2, 1.0, 0.9, 0.8, 0.9, 1.1, 1.2, 1.0, 0.9]
        },
        'Beans': {
            'seed': [1.2, 1.3, 1.4, 1.2, 1.0, 0.9, 0.8, 0.9, 1.1, 1.3, 1.2, 1.0],
            'fertilizer': [1.1, 1.2, 1.3, 1.2, 1.0, 0.9, 0.8, 0.9, 1.2, 1.3, 1.1, 1.0],
            'herbicide': [0.9, 1.0, 1.2, 1.3, 1.1, 0.9, 0.8, 0.9, 1.1, 1.2, 1.0, 0.9],
            'pesticide': [0.9, 1.1, 1.3, 1.2, 1.0, 0.9, 0.8, 0.9, 1.1, 1.2, 1.0, 0.9],
            'labor': [1.0, 1.1, 1.3, 1.2, 1.0, 0.9, 0.8, 0.9, 1.1, 1.2, 1.0, 0.9]
        },
        'Cassava': {
            'seed': [0.9, 1.0, 1.1, 1.2, 1.1, 0.9, 0.8, 0.9, 1.0, 1.1, 1.0, 0.9],
            'fertilizer': [0.9, 1.0, 1.2, 1.1, 1.0, 0.9, 0.8, 0.9, 1.1, 1.0, 0.9, 0.9],
            'herbicide': [0.8, 0.9, 1.1, 1.2, 1.0, 0.9, 0.8, 0.9, 1.0, 1.1, 0.9, 0.8],
            'pesticide': [0.9, 1.0, 1.1, 1.2, 1.0, 0.9, 0.8, 0.9, 1.0, 1.1, 0.9, 0.9],
            'labor': [0.9, 1.0, 1.2, 1.1, 1.0, 0.9, 0.8, 0.9, 1.0, 1.1, 1.0, 0.9]
        },
        'Rice': {
            'seed': [1.1, 1.2, 1.3, 1.2, 1.0, 0.9, 0.8, 0.9, 1.1, 1.2, 1.1, 1.0],
            'fertilizer': [1.2, 1.3, 1.4, 1.2, 1.0, 0.9, 0.8, 0.9, 1.2, 1.3, 1.1, 1.0],
            'herbicide': [0.9, 1.1, 1.3, 1.4, 1.1, 0.9, 0.8, 0.9, 1.1, 1.3, 1.0, 0.9],
            'pesticide': [1.0, 1.1, 1.3, 1.3, 1.1, 0.9, 0.8, 0.9, 1.2, 1.3, 1.1, 1.0],
            'labor': [1.0, 1.2, 1.3, 1.3, 1.1, 0.9, 0.8, 0.9, 1.2, 1.2, 1.1, 1.0]
        }
    }
}

MARKET_TRENDS = {
    'seed': 0.018,
    'fertilizer': 0.025,
    'herbicide': 0.020,
    'pesticide': 0.019,
    'labor': 0.015
}

def create_batch_edges(batch_size, nodes_per_graph=4, device='cpu'):
    """Create edge indices for batched graphs - returns None if not using GAT"""
    try:
        edge_indices = []
        for b in range(batch_size):
            offset = b * nodes_per_graph
            for i in range(nodes_per_graph):
                for j in range(nodes_per_graph):
                    if i != j:
                        edge_indices.append([i + offset, j + offset])
        return torch.tensor(edge_indices, dtype=torch.long).t().to(device)
    except:
        return None

def prepare_input_features(dataset, scalers, current_date, region, district, crop):
    month = current_date.month
    year = current_date.year - 2020
    
    temporal_features = np.concatenate([
        dataset.data.iloc[-1, dataset.price_cols].values.astype(float),
        [month / 12],
        [year / 5]
    ]).reshape(1, -1)
    
    graph_features = np.concatenate([
        [dataset.encoders[col].transform([locals()[col.lower()]])[0] 
         for col in dataset.categorical_cols],
        dataset.data.iloc[-1, dataset.price_cols].values.astype(float)
    ]).reshape(1, -1)
    
    return temporal_features, graph_features

def download_model_if_needed():
    """Download model from external storage if not present"""
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pth')
    
    if not os.path.exists(model_path):
        print("Model not found locally, downloading from external storage...")
        
        # Replace with your actual URL (Google Drive, Dropbox, GitHub Releases, etc.)
        MODEL_URL = os.environ.get('MODEL_URL', 'YOUR_MODEL_URL_HERE')
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        response = requests.get(MODEL_URL, stream=True)
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Model downloaded successfully!")
    
    return model_path

def load_model_and_dataset():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    test_csv = os.path.join(base_path, 'test_dataset_cleaned.csv')
    train_csv = os.path.join(base_path, 'train_dataset_cleaned.csv')
    model_path = os.path.join(base_path, 'models', 'gat_rnn_hybrid_best.pth')
    
    print(f"📂 Base path: {base_path}")
    print(f"📄 Looking for test CSV: {test_csv}")
    print(f"📄 Looking for train CSV: {train_csv}")
    print(f"🤖 Looking for model: {model_path}")
    
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test dataset not found at {test_csv}")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train dataset not found at {train_csv}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print("✅ All files found!")
    
    test_df = pd.read_csv(test_csv)
    print(f"✅ Test data loaded: {len(test_df)} rows")
    
    train_df = pd.read_csv(train_csv)
    print(f"✅ Train data loaded: {len(train_df)} rows")
    
    train_dataset = AgriculturalDataset(train_df, training=True)
    
    test_dataset = AgriculturalDataset(
        test_df,
        encoders=train_dataset.encoders,tate_dict'])
        scalers=train_dataset.scalers
    )    print("✅ Loading from state dict")
    dict(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}")
    aded and set to eval mode!")
    # Initialize model with correct hidden_size from checkpoint (32, not 8)
    print("🏗️ Initializing model architecture..."), device
    model = GATRNNHybrid(
        temporal_input_size=7,ice, num_samples=5):
        graph_input_size=8,es = ['Seed', 'Fertilizer', 'Herbicide', 'Pesticide', 'Labor']
        hidden_size=32
    ).to(device)
    nPredicting Individual Input Costs:")
    # Load checkpoint
    print(f"📦 Loading model weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:if i >= num_samples:
        print(f"✅ Loading from full checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    print("\nPredicting Individual Input Costs:")
    print("-" * 50)tch['temporal'].to(device)
    graph_x = batch['graph'].to(device)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_samples:
                breakvice=device)
                
            temporal_x = batch['temporal'].to(device)            pred = model(temporal_x, graph_x, edge_index, batch_idx)
            graph_x = batch['graph'].to(device)
            actual = batch['target'].to(device)_prices = scalers['target'].inverse_transform(pred.cpu())
            rs['target'].inverse_transform(actual.cpu())
            edge_index = create_batch_edges(1, device=device)
            batch_idx = torch.zeros(4, dtype=torch.long, device=device)        print(f"\nSample {i+1}:")
            >10} {'Diff %':>10}")
            pred = model(temporal_x, graph_x, edge_index, batch_idx))
            
            pred_prices = scalers['target'].inverse_transform(pred.cpu()), actual) in enumerate(zip(cost_names, 
            actual_prices = scalers['target'].inverse_transform(actual.cpu()), 
                                                               actual_prices[0])):
            print(f"\nSample {i+1}:")
            print(f"{'Input Cost':<12} {'Predicted':>10} {'Actual':>10} {'Diff %':>10}")            print(f"{name:<12} {pred:>10.2f} {actual:>10.2f} {diff_pct:>9.1f}%")
            print("-" * 45)
            , scalers, device, region, district, crop, acres=1, forecast_months=6):
            for j, (name, pred, actual) in enumerate(zip(cost_names, 
                                                       pred_prices[0], 
                                                       actual_prices[0])):ons = []
                diff_pct = ((pred - actual) / actual) * 100
                print(f"{name:<12} {pred:>10.2f} {actual:>10.2f} {diff_pct:>9.1f}%")
op_seasonals:
def predict_future_prices(model, dataset, scalers, device, region, district, crop, acres=1, forecast_months=6):_seasonals = SEASONAL_MULTIPLIERS.get('Central', {}).get(crop)
    model.eval()
    current_date = datetime.now()ize']
    predictions = []
    _inflation_mult = REGIONAL_INFLATION.get(region, 1.0)
    crop_seasonals = SEASONAL_MULTIPLIERS.get(region, {}).get(crop)
    if not crop_seasonals:
        crop_seasonals = SEASONAL_MULTIPLIERS.get('Central', {}).get(crop)s):
        if not crop_seasonals:future_date = current_date + relativedelta(months=i)
            crop_seasonals = SEASONAL_MULTIPLIERS['Central']['Maize']
    
    regional_inflation_mult = REGIONAL_INFLATION.get(region, 1.0)temporal_features, graph_features = prepare_input_features(
    p
    with torch.no_grad():
        for i in range(forecast_months):
            future_date = current_date + relativedelta(months=i)
            month_idx = future_date.month - 1    scalers['temporal'].transform(temporal_features)
            
            temporal_features, graph_features = prepare_input_features(
                dataset, scalers, future_date, region, district, crop
            )    scalers['graph'].transform(graph_features)
            
            temporal_x = torch.FloatTensor(
                scalers['temporal'].transform(temporal_features)
            ).to(device)batch_idx = torch.zeros(4, dtype=torch.long, device=device)
            
            graph_x = torch.FloatTensor(x, batch_idx)
                scalers['graph'].transform(graph_features)
            ).repeat(4, 1).to(device)
            
            edge_index = create_batch_edges(1, device=device)
            batch_idx = torch.zeros(4, dtype=torch.long, device=device)
            
            pred = model(temporal_x, graph_x, edge_index, batch_idx)
            base_prices = scalers['target'].inverse_transform(pred.cpu().numpy())[0]
            size=5)
            compound_inflation = (1 + INFLATION_RATE * regional_inflation_mult) ** i
            arket_noise = np.random.uniform(0.98, 1.02, size=5)
            trend_factors = {k: (1 + v) ** i for k, v in MARKET_TRENDS.items()}
            
            weather_factor = WEATHER_RISK.get(future_date.month, 1.0)            'month': future_date.strftime('%B %Y'),
            : float(base_prices[0] * crop_seasonals['seed'][month_idx] * trend_factors['seed'] * 
            supply_chain_factor = np.random.uniform(1 - SUPPLY_CHAIN_VOLATILITY, 1 + SUPPLY_CHAIN_VOLATILITY, size=5)                            compound_inflation * weather_factor * supply_chain_factor[0] * market_noise[0]),
            r': float(base_prices[1] * crop_seasonals['fertilizer'][month_idx] * trend_factors['fertilizer'] * 
            market_noise = np.random.uniform(0.98, 1.02, size=5)_factor * supply_chain_factor[1] * market_noise[1]),
                        'herbicide': float(base_prices[2] * crop_seasonals['herbicide'][month_idx] * trend_factors['herbicide'] * 
            adjusted_prices = {und_inflation * weather_factor * supply_chain_factor[2] * market_noise[2]),
                'month': future_date.strftime('%B %Y'),prices[3] * crop_seasonals['pesticide'][month_idx] * trend_factors['pesticide'] * 
                'seed': float(base_prices[0] * crop_seasonals['seed'][month_idx] * trend_factors['seed'] *         compound_inflation * weather_factor * supply_chain_factor[3] * market_noise[3]),
                            compound_inflation * weather_factor * supply_chain_factor[0] * market_noise[0]),loat(base_prices[4] * crop_seasonals['labor'][month_idx] * trend_factors['labor'] * 
                'fertilizer': float(base_prices[1] * crop_seasonals['fertilizer'][month_idx] * trend_factors['fertilizer'] *        compound_inflation * weather_factor * supply_chain_factor[4] * market_noise[4])
                                  compound_inflation * weather_factor * supply_chain_factor[1] * market_noise[1]),
                'herbicide': float(base_prices[2] * crop_seasonals['herbicide'][month_idx] * trend_factors['herbicide'] *        
                                 compound_inflation * weather_factor * supply_chain_factor[2] * market_noise[2]),        predictions.append(adjusted_prices)
                'pesticide': float(base_prices[3] * crop_seasonals['pesticide'][month_idx] * trend_factors['pesticide'] * 
                                 compound_inflation * weather_factor * supply_chain_factor[3] * market_noise[3]),ons
                'labor': float(base_prices[4] * crop_seasonals['labor'][month_idx] * trend_factors['labor'] * 
                             compound_inflation * weather_factor * supply_chain_factor[4] * market_noise[4])
            } scalers, device = load_model_and_dataset()
            
            predictions.append(adjusted_prices)ture_prices(
    
    return predictions
asaka',
if __name__ == '__main__':
    model, dataset, scalers, device = load_model_and_dataset()
    
    forecasts = predict_future_prices(
        model, dataset, scalers, device,ts:")
        region='Central',
        district='Masaka',
        crop='Coffee',          f"{'Pesticide':>12} {'Labor':>10}")





















              f"{pred['herbicide']:>12.0f} {pred['pesticide']:>12.0f} {pred['labor']:>10.0f}")        print(f"{pred['month']:<12} {pred['seed']:>10.0f} {pred['fertilizer']:>12.0f} "    for pred in forecasts:        print("-" * 70)          f"{'Pesticide':>12} {'Labor':>10}")    print(f"{'Month':<12} {'Seed':>10} {'Fertilizer':>12} {'Herbicide':>12} "    print("-" * 70)              f"{pred['herbicide']:>12.0f} {pred['pesticide']:>12.0f} {pred['labor']:>10.0f}")        print(f"{pred['month']:<12} {pred['seed']:>10.0f} {pred['fertilizer']:>12.0f} "    for pred in forecasts:        print("-" * 70)          f"{'Pesticide':>12} {'Labor':>10}")    print(f"{'Month':<12} {'Seed':>10} {'Fertilizer':>12} {'Herbicide':>12} "    print("-" * 70)    print("\nPrice Forecasts:")        )        forecast_months=6    print("-" * 70)
    
    for pred in forecasts:
        print(f"{pred['month']:<12} {pred['seed']:>10.0f} {pred['fertilizer']:>12.0f} "
              f"{pred['herbicide']:>12.0f} {pred['pesticide']:>12.0f} {pred['labor']:>10.0f}")
    print("-" * 70)
    print(f"{'Month':<12} {'Seed':>10} {'Fertilizer':>12} {'Herbicide':>12} "
          f"{'Pesticide':>12} {'Labor':>10}")
    print("-" * 70)
    
    for pred in forecasts:
        print(f"{pred['month']:<12} {pred['seed']:>10.0f} {pred['fertilizer']:>12.0f} "
              f"{pred['herbicide']:>12.0f} {pred['pesticide']:>12.0f} {pred['labor']:>10.0f}")
