import gradio as gr
import json
from predict import load_model_and_dataset, predict_future_prices

print("Loading model and dataset...")
model, dataset, scalers, device = load_model_and_dataset()
print("Model loaded successfully!")

def predict_prices(region, district, crop, acres, forecast_months):
    """Make price predictions"""
    try:
        predictions = predict_future_prices(
            model, dataset, scalers, device,
            region=region,
            district=district,
            crop=crop,
            acres=float(acres),
            forecast_months=int(forecast_months)
        )
        
        # Format output as table
        output = f"# Price Forecasts for {crop} in {district}, {region}\n\n"
        output += f"**Farm Size:** {acres} acres | **Forecast Period:** {forecast_months} months\n\n"
        output += "| Month | Seed (UGX) | Fertilizer (UGX) | Herbicide (UGX) | Pesticide (UGX) | Labor (UGX) |\n"
        output += "|-------|------------|------------------|-----------------|-----------------|-------------|\n"
        
        for pred in predictions:
            output += f"| {pred['month']} | {pred['seed']:,.0f} | {pred['fertilizer']:,.0f} | {pred['herbicide']:,.0f} | {pred['pesticide']:,.0f} | {pred['labor']:,.0f} |\n"
        
        return output
        
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Agricultural Price Predictor - Uganda") as demo:
    gr.Markdown("# 🌾 Agricultural Input Price Predictor")
    gr.Markdown("Predict prices for agricultural inputs in Uganda using AI")
    
    with gr.Row():
        with gr.Column():
            region = gr.Dropdown(
                ["Central", "Eastern", "Western", "Northern"],
                label="Region",
                value="Central"
            )
            district = gr.Textbox(label="District", value="Kampala")
            crop = gr.Dropdown(
                ["Maize", "Coffee", "Sweet Potatoes"],
                label="Crop Type",
                value="Maize"
            )
            acres = gr.Number(label="Farm Size (Acres)", value=1, minimum=0.1)
            forecast_months = gr.Slider(
                1, 12,
                value=6,
                step=1,
                label="Forecast Months"
            )
            predict_btn = gr.Button("Predict Prices", variant="primary")
        
        with gr.Column():
            output = gr.Markdown(label="Price Forecasts")
    
    predict_btn.click(
        fn=predict_prices,
        inputs=[region, district, crop, acres, forecast_months],
        outputs=output
    )
    
    gr.Markdown("""
    ### About
    This tool uses machine learning to forecast agricultural input prices in Uganda.
    Predictions are based on historical data, seasonal patterns, and regional factors.
    
    **Input Costs Predicted:**
    - Seeds
    - Fertilizers
    - Herbicides
    - Pesticides
    - Labor
    """)

if __name__ == "__main__":
    demo.launch()
