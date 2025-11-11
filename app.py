import gradio as gr
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("="*50)
print("Starting Agricultural Price Predictor")
print("="*50)

MODEL_LOADED = False
model, encoders, scaler_temporal, scaler_graph, scalers_y, price_cols, device = None, None, None, None, None, None, None

try:
    from predict import load_model_and_dataset, predict_future_prices
    model, encoders, scaler_temporal, scaler_graph, scalers_y, price_cols, device = load_model_and_dataset()
    MODEL_LOADED = True
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(traceback.format_exc())

def predict_prices(region, district, crop, acres, forecast_months):
    """Make price predictions"""
    if not MODEL_LOADED:
        return "⚠️ **Model Loading Error**\n\nThe AI model failed to load. Check container logs."
    
    try:
        predictions = predict_future_prices(
            model, encoders, scaler_temporal, scaler_graph, scalers_y, price_cols, device,
            region=region,
            district=district,
            crop=crop,
            acres=float(acres),
            forecast_months=int(forecast_months)
        )
        
        output = f"# 📊 Price Forecasts for {crop} in {district}, {region}\n\n"
        output += f"**Farm Size:** {acres} acres | **Forecast Period:** {forecast_months} months\n\n"
        output += f"**Model Performance:** R² = 0.86, MAPE = 13-15%\n\n"
        output += "| Month | Seed (UGX) | Fertilizer (UGX) | Herbicide (UGX) | Pesticide (UGX) | Labor (UGX) |\n"
        output += "|-------|------------|------------------|-----------------|-----------------|-------------|\n"
        
        for pred in predictions:
            output += f"| {pred['month']} | {pred['seed']:,.0f} | {pred['fertilizer']:,.0f} | {pred['herbicide']:,.0f} | {pred['pesticide']:,.0f} | {pred['labor']:,.0f} |\n"
        
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
        output += f"\n---\n*Powered by GAT-RNN Hybrid AI (3-Layer Graph Attention + Bidirectional LSTM)*"
        
        return output
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return f"❌ **Error:** {str(e)}\n\nPlease try different inputs."

def update_districts(region):
    """Update district dropdown based on selected region"""
    districts_by_region = {
        'Central': ['Kampala', 'Wakiso', 'Mukono', 'Masaka', 'Mubende', 'Mityana'],
        'Eastern': ['Mbale', 'Jinja', 'Soroti', 'Tororo', 'Iganga', 'Kamuli'],
        'Western': ['Mbarara', 'Kabale', 'Fort Portal', 'Kasese', 'Hoima'],
        'Northern': ['Gulu', 'Lira', 'Arua', 'Kitgum', 'Pader']
    }
    choices = districts_by_region.get(region, [])
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Agricultural Price Predictor - Uganda") as demo:
    gr.Markdown("""
    # 🌾 Agricultural Input Price Predictor
    ### AI-Powered Price Forecasting for Ugandan Farmers
    
    Predict future prices for agricultural inputs using a **GAT-RNN Hybrid AI model** (R² = 0.86, Accuracy: 85-87%).
    Get accurate forecasts for seeds, fertilizers, herbicides, pesticides, and labor costs.
    
    **Model:** 3-Layer Graph Attention Network + Bidirectional LSTM
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
                choices=["Kampala", "Wakiso", "Mukono", "Masaka", "Mubende", "Mityana"],
                label="District",
                value="Kampala",
                info="Select your district"
            )
            
            # Update districts when region changes
            region.change(fn=update_districts, inputs=region, outputs=district)
            
            crop = gr.Dropdown(
                choices=["Maize", "Coffee", "Sweet Potatoes", "Beans", "Cassava", "Rice"],
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
            
            # Show model status
            if MODEL_LOADED:
                status_msg = "✅ **Model Status:** Loaded (R² = 0.86, MAPE = 13-15%)"
            else:
                status_msg = "❌ **Model Status:** Failed to load"
            
            gr.Markdown(status_msg)
            output = gr.Markdown(value="*Click 'Predict Prices' to see forecasts*")
    
    predict_btn.click(
        fn=predict_prices,
        inputs=[region, district, crop, acres, forecast_months],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### 📚 About This Tool
    
    This AI model uses:
    - **Graph Attention Networks (GAT)** to capture spatial relationships between regions, districts, and crops
    - **Bidirectional LSTM** to learn temporal price patterns
    - **Multi-layer architecture** with 578k parameters for accurate predictions
    
    **Performance:** R² = 0.86 (86% accuracy), Average error: 13-15%
    
    Built with PyTorch Geometric and deployed on Hugging Face Spaces.
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
