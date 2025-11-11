import gradio as gr
import json
import os
import logging
from predict import load_model_and_dataset, predict_future_prices

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("Loading model and dataset...")
try:
    model, dataset, scalers, device = load_model_and_dataset()
    MODEL_LOADED = True
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL_LOADED = False
    model, dataset, scalers, device = None, None, None, None

def predict_prices(region, district, crop, acres, forecast_months):
    """Make price predictions"""
    if not MODEL_LOADED:
        return "⚠️ **Error:** Model not loaded. Please check logs."
    
    try:
        predictions = predict_future_prices(
            model, dataset, scalers, device,
            region=region,
            district=district,
            crop=crop,
            acres=float(acres),
            forecast_months=int(forecast_months)
        )
        
        # Format output as markdown table
        output = f"# 📊 Price Forecasts for {crop} in {district}, {region}\n\n"
        output += f"**Farm Size:** {acres} acres | **Forecast Period:** {forecast_months} months\n\n"
        output += "| Month | Seed (UGX) | Fertilizer (UGX) | Herbicide (UGX) | Pesticide (UGX) | Labor (UGX) |\n"
        output += "|-------|------------|------------------|-----------------|-----------------|-------------|\n"
        
        for pred in predictions:
            output += f"| {pred['month']} | {pred['seed']:,.0f} | {pred['fertilizer']:,.0f} | {pred['herbicide']:,.0f} | {pred['pesticide']:,.0f} | {pred['labor']:,.0f} |\n"
        
        # Add total costs
        total = {
            'seed': sum(p['seed'] for p in predictions),
            'fertilizer': sum(p['fertilizer'] for p in predictions),
            'herbicide': sum(p['herbicide'] for p in predictions),
            'pesticide': sum(p['pesticide'] for p in predictions),
            'labor': sum(p['labor'] for p in predictions)
        }
        
        output += f"\n### 💰 Total Estimated Costs ({forecast_months} months)\n\n"
        output += f"- **Seeds:** {total['seed']:,.0f} UGX\n"
        output += f"- **Fertilizer:** {total['fertilizer']:,.0f} UGX\n"
        output += f"- **Herbicide:** {total['herbicide']:,.0f} UGX\n"
        output += f"- **Pesticide:** {total['pesticide']:,.0f} UGX\n"
        output += f"- **Labor:** {total['labor']:,.0f} UGX\n"
        output += f"\n**Grand Total:** {sum(total.values()):,.0f} UGX\n"
        
        return output
        
    except Exception as e:
        return f"❌ **Error:** {str(e)}\n\nPlease try different inputs or contact support."

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Agricultural Price Predictor - Uganda") as demo:
    gr.Markdown("""
    # 🌾 Agricultural Input Price Predictor
    ### AI-Powered Price Forecasting for Ugandan Farmers
    
    Predict future prices for agricultural inputs using machine learning. 
    Get accurate forecasts for seeds, fertilizers, herbicides, pesticides, and labor costs.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📍 Location & Crop Details")
            
            region = gr.Dropdown(
                choices=["Central", "Eastern", "Western", "Northern"],
                label="Region",
                value="Central",
                info="Select your farming region"
            )
            
            district = gr.Dropdown(
                choices=["Kampala", "Wakiso", "Mukono", "Masaka", "Mbale", "Jinja", 
                        "Soroti", "Mbarara", "Kabale", "Gulu", "Lira", "Arua"],
                label="District",
                value="Kampala",
                info="Select your district"
            )
            
            crop = gr.Dropdown(
                choices=["Maize", "Coffee", "Sweet Potatoes"],
                label="Crop Type",
                value="Maize",
                info="What are you growing?"
            )
            
            gr.Markdown("### 📏 Farm & Forecast Settings")
            
            acres = gr.Number(
                label="Farm Size (Acres)",
                value=1,
                minimum=0.1,
                maximum=1000,
                info="Enter your farm size"
            )
            
            forecast_months = gr.Slider(
                minimum=1,
                maximum=12,
                value=6,
                step=1,
                label="Forecast Period (Months)",
                info="How many months to forecast?"
            )
            
            predict_btn = gr.Button("🔮 Predict Prices", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### 📈 Price Forecasts")
            output = gr.Markdown(value="*Click 'Predict Prices' to see forecasts*")
    
    predict_btn.click(
        fn=predict_prices,
        inputs=[region, district, crop, acres, forecast_months],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### ℹ️ About This Tool
    
    This AI-powered tool uses a hybrid **Graph Attention Network (GAT)** and **Recurrent Neural Network (RNN)** 
    to forecast agricultural input prices in Uganda. The model considers:
    
    - 📊 Historical price trends
    - 🌦️ Seasonal variations
    - 📍 Regional differences
    - 💹 Market dynamics
    - 🌾 Crop-specific patterns
    
    **Input Costs Predicted:**
    - 🌱 Seeds
    - 🧪 Fertilizers
    - 🌿 Herbicides
    - 🐛 Pesticides
    - 👷 Labor
    
    **Data Source:** Historical agricultural market data from Uganda (2020-2024)
    
    **Note:** Predictions are estimates based on historical patterns. Actual prices may vary due to 
    unforeseen market conditions, weather events, or policy changes.
    
    ---
    
    💡 **Tip:** Use these forecasts to plan your farming budget and make informed purchasing decisions!
    
    🔗 **GitHub:** [otimongjonathan/newImproved](https://github.com/otimongjonathan/newImproved)
    
    Made with ❤️ for Ugandan farmers
    """)

if __name__ == "__main__":
    demo.launch()
