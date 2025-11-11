import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from app.services.prediction_service import PredictionService
import torch
import os
import json

def validate_model():
    print("=" * 60)
    print("Model Validation Report")
    print("=" * 60)

    # Load datasets
    train_df = pd.read_csv('train_dataset_cleaned.csv')
    test_df = pd.read_csv('test_dataset_cleaned.csv')
    val_df = pd.read_csv('validation_dataset_cleaned.csv')

    # Initialize prediction service
    prediction_service = PredictionService()
    prediction_service.load_model_from_path(
        'models/hybrid_agricultural_model_best.pth',
        'models/preprocessor.pkl'
    )

    # Validate on test set
    predictions = []
    actuals = []

    for _, row in test_df.iterrows():
        try:
            pred = prediction_service.predict_individual_costs(
                row['Region'], 
                row['District'],
                row['Crop']
            )
            
            predictions.append([
                pred['seed_price_per_kg'],
                pred['fertilizer_price_per_kg'],
                pred['herbicide_price_per_litre'],
                pred['pesticide_price_per_litre'],
                pred['labor_cost_per_day']
            ])
            
            actuals.append([
                row['Seed_Price'],
                row['Fertilizer_Price'],
                row['Herbicide_Price'],
                row['Pesticide_Price'],
                row['Labor_Cost']
            ])
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    metrics = {
        'r2_scores': {},
        'rmse_scores': {},
        'mae_scores': {}
    }

    input_types = ['Seed', 'Fertilizer', 'Herbicide', 'Pesticide', 'Labor']
    
    for i, input_type in enumerate(input_types):
        metrics['r2_scores'][input_type] = r2_score(actuals[:, i], predictions[:, i])
        metrics['rmse_scores'][input_type] = np.sqrt(mean_squared_error(actuals[:, i], predictions[:, i]))
        metrics['mae_scores'][input_type] = mean_absolute_error(actuals[:, i], predictions[:, i])

    # Print results
    print("\nModel Performance Metrics:")
    print("-" * 40)
    
    for input_type in input_types:
        print(f"\n{input_type}:")
        print(f"R² Score: {metrics['r2_scores'][input_type]:.4f}")
        print(f"RMSE: {metrics['rmse_scores'][input_type]:.2f} UGX")
        print(f"MAE: {metrics['mae_scores'][input_type]:.2f} UGX")

    # Calculate total cost metrics
    total_pred = predictions.sum(axis=1)
    total_actual = actuals.sum(axis=1)
    
    total_r2 = r2_score(total_actual, total_pred)
    total_rmse = np.sqrt(mean_squared_error(total_actual, total_pred))
    total_mae = mean_absolute_error(total_actual, total_pred)

    print("\nTotal Cost Prediction:")
    print(f"R² Score: {total_r2:.4f}")
    print(f"RMSE: {total_rmse:.2f} UGX")
    print(f"MAE: {total_mae:.2f} UGX")

    # Save metrics
    metrics['total_metrics'] = {
        'r2': total_r2,
        'rmse': total_rmse,
        'mae': total_mae
    }

    with open('models/validation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nDeployment Readiness Assessment:")
    print("-" * 40)
    
    # Assess model readiness
    is_ready = True
    warnings = []

    if total_r2 < 0.7:
        is_ready = False
        warnings.append("❌ Total cost R² score below 0.7")
    else:
        print("✓ Total cost R² score acceptable")

    if total_rmse > 50000:  # 50,000 UGX threshold
        is_ready = False
        warnings.append("❌ Total cost RMSE too high")
    else:
        print("✓ Total cost RMSE acceptable")

    for input_type in input_types:
        if metrics['r2_scores'][input_type] < 0.65:
            warnings.append(f"⚠ Low R² score for {input_type}")

    if is_ready:
        print("\n✅ Model is READY for deployment!")
    else:
        print("\n❌ Model needs improvement before deployment")
    
    if warnings:
        print("\nWarnings:")
        for warning in warnings
        print("✓ Validation report saved to models/validation_metrics.json")

if __name__ == '__main__':
    validate_model()
