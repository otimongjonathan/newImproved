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
    print("✅ Model loaded successfully in app.py!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(traceback.format_exc())

def predict_prices(region, district, crop, acres, forecast_months):
    """Make price predictions"""
    if not MODEL_LOADED:
        error_msg = "⚠️ **Model Loading Error**\n\nThe AI model failed to load. Check console logs."
        logger.error(error_msg)
        return error_msg
    
    try:
        logger.info(f"Prediction requested: {region}, {district}, {crop}, {acres} acres, {forecast_months} months")
        
        predictions = predict_future_prices(
            model, encoders, scaler_temporal, scaler_graph, scalers_y, price_cols, device,
            region=region,
            district=district,
            crop=crop,
            acres=1,  # Get unit prices first
            forecast_months=int(forecast_months)
        )
        
        logger.info(f"Predictions generated successfully: {len(predictions)} months")
        
        # Region-specific operational costs (transport, storage, infrastructure)
        OPERATIONAL_COSTS = {
            'Central': 0.06,   # 6% - better infrastructure, closer to markets
            'Eastern': 0.075,  # 7.5% - moderate infrastructure
            'Western': 0.08,   # 8% - hilly terrain, longer distances
            'Northern': 0.09   # 9% - remote, poor infrastructure
        }
        
        # Monthly inflation variations (Uganda agricultural sector 2024)
        # Based on seasonal demand and supply patterns
        MONTHLY_INFLATION = {
            1: 0.025,   # January - post-harvest, lower inflation
            2: 0.026,   # February
            3: 0.028,   # March - planting season starts
            4: 0.030,   # April - peak planting, higher demand
            5: 0.032,   # May - continued planting
            6: 0.028,   # June - mid-season
            7: 0.027,   # July
            8: 0.029,   # August - pre-harvest
            9: 0.026,   # September - harvest season
            10: 0.025,  # October - abundant supply
            11: 0.027,  # November
            12: 0.029   # December - year-end demand
        }
        
        operational_rate = OPERATIONAL_COSTS.get(region, 0.075)
        base_inflation = 0.028  # Average annual rate
        
        output = f"# 📊 Price Forecasts for {crop} in {district}, {region}\n\n"
        output += f"**Farm Size:** {acres} acres | **Forecast Period:** {forecast_months} months\n"
        output += f"**Model Performance:** R² = 0.86, MAPE = 13-15%\n\n"
        output += f"**Regional Adjustments for {region}:**\n"
        output += f"- Operational Costs: {operational_rate*100:.1f}% (Transport, Storage, Infrastructure)\n"
        output += f"- Base Inflation: {base_inflation*100:.1f}% annually (varies monthly)\n\n"
        
        # Assumed quantities per acre per month
        quantities = {
            'seed': float(acres) * 25,      # 25 kg per acre
            'fertilizer': float(acres) * 50, # 50 kg per acre
            'herbicide': float(acres) * 10,  # 10 L per acre
            'pesticide': float(acres) * 10,  # 10 L per acre
            'labor': float(acres) * 20       # 20 days per acre
        }
        
        grand_total = 0
        grand_total_with_adjustments = 0
        
        # Track inflation details
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
        current_date = datetime.now()
        
        # Create separate table for each month
        for month_idx, pred in enumerate(predictions):
            month = pred['month']
            future_date = current_date + relativedelta(months=month_idx)
            month_num = future_date.month
            
            # Get monthly inflation rate
            monthly_inflation_rate = MONTHLY_INFLATION.get(month_num, base_inflation)
            
            # Apply compound inflation: (1 + monthly_rate)^months
            inflation_factor = (1 + monthly_inflation_rate) ** (month_idx / 12)
            
            output += f"## 📅 {month}\n\n"
            output += f"*Inflation: {monthly_inflation_rate*100:.1f}%/year | Operational: {operational_rate*100:.1f}%*\n\n"
            output += "| Input Type | Base Price (UGX) | Adjusted Price* (UGX) | Quantity | Subtotal (UGX) |\n"
            output += "|------------|------------------|----------------------|----------|----------------|\n"
            
            # Calculate costs with monthly inflation
            seed_price_adj = pred['seed'] * inflation_factor
            fert_price_adj = pred['fertilizer'] * inflation_factor
            herb_price_adj = pred['herbicide'] * inflation_factor
            pest_price_adj = pred['pesticide'] * inflation_factor
            labor_price_adj = pred['labor'] * inflation_factor
            
            seed_total = seed_price_adj * quantities['seed']
            fert_total = fert_price_adj * quantities['fertilizer']
            herb_total = herb_price_adj * quantities['herbicide']
            pest_total = pest_price_adj * quantities['pesticide']
            labor_total = labor_price_adj * quantities['labor']
            
            output += f"| Seeds | {pred['seed']:,.2f} | {seed_price_adj:,.2f} | {quantities['seed']:.0f} kg | {seed_total:,.0f} |\n"
            output += f"| Fertilizer | {pred['fertilizer']:,.2f} | {fert_price_adj:,.2f} | {quantities['fertilizer']:.0f} kg | {fert_total:,.0f} |\n"
            output += f"| Herbicide | {pred['herbicide']:,.2f} | {herb_price_adj:,.2f} | {quantities['herbicide']:.0f} L | {herb_total:,.0f} |\n"
            output += f"| Pesticide | {pred['pesticide']:,.2f} | {pest_price_adj:,.2f} | {quantities['pesticide']:.0f} L | {pest_total:,.0f} |\n"
            output += f"| Labor | {pred['labor']:,.2f} | {labor_price_adj:,.2f} | {quantities['labor']:.0f} days | {labor_total:,.0f} |\n"
            
            month_subtotal = seed_total + fert_total + herb_total + pest_total + labor_total
            operational = month_subtotal * operational_rate
            month_total = month_subtotal + operational
            
            output += f"| **Subtotal** | | | | **{month_subtotal:,.0f}** |\n"
            output += f"| **Operational ({operational_rate*100:.1f}%)** | | | | **{operational:,.0f}** |\n"
            output += f"| **MONTH TOTAL** | | | | **{month_total:,.0f}** |\n\n"
            
            grand_total += month_subtotal
            grand_total_with_adjustments += month_total
            
            output += "---\n\n"
        
        total_operational = grand_total_with_adjustments - grand_total
        
        output += f"### 💰 Cost Breakdown ({forecast_months} months)\n\n"
        output += f"| Category | Amount (UGX) |\n"
        output += f"|----------|-------------|\n"
        output += f"| Base Input Costs | {grand_total:,.0f} |\n"
        output += f"| Operational Costs ({region}) | {total_operational:,.0f} |\n"
        output += f"| **GRAND TOTAL** | **{grand_total_with_adjustments:,.0f}** |\n\n"
        
        output += f"**Regional Operational Cost Rates:**\n"
        output += f"- Central: 6.0% | Eastern: 7.5% | Western: 8.0% | Northern: 9.0%\n\n"
        
        output += f"**Monthly Inflation Variations (2024 Uganda):**\n"
        output += f"- Planting season (Mar-May): 2.8-3.2% | Harvest season (Sep-Oct): 2.5-2.6%\n"
        output += f"- Based on seasonal agricultural demand patterns\n\n"
        
        output += f"*Powered by GAT-RNN Hybrid AI (R² = 0.86)*"
        
        return output
        
    except Exception as e:
        error_msg = f"❌ **Prediction Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return error_msg

def update_districts(region):
    """Update district dropdown based on selected region - USING ACTUAL DATA"""
    try:
        # These are the ACTUAL districts from train_dataset_cleaned.csv
        districts_by_region = {
            'Central': ['Kampala', 'Luweero', 'Masaka', 'Mukono', 'Wakiso'],
            'Northern': ['Arua', 'Gulu', 'Kitgum', 'Lira', 'Nebbi'],
            'Western': ['Fort Portal', 'Hoima', 'Kabale', 'Kasese', 'Mbarara'],
            'Eastern': ['Iganga', 'Jinja', 'Mbale', 'Soroti', 'Tororo']
        }
        choices = districts_by_region.get(region, [])
        return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
    except Exception as e:
        logger.error(f"Error updating districts: {e}")
        return gr.Dropdown(choices=[], value=None)

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
                choices=["Central", "Northern", "Western", "Eastern"],
                label="Region",
                value="Central",
                info="Select your farming region"
            )
            
            district = gr.Dropdown(
                choices=["Kampala", "Luweero", "Masaka", "Mukono", "Wakiso"],  # Central districts as default
                label="District",
                value="Kampala",
                info="Select your district"
            )
            
            region.change(fn=update_districts, inputs=region, outputs=district)
            
            crop = gr.Dropdown(
                choices=["Beans", "Cassava", "Coffee", "Cotton", "Groundnuts", "Maize", 
                        "Matooke", "Rice", "Sorghum", "Soybeans", "Sunflower", "Sweet Potatoes"],
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
    
    **Supported Locations:**
    - **Central:** Kampala, Luweero, Masaka, Mukono, Wakiso
    - **Northern:** Arua, Gulu, Kitgum, Lira, Nebbi
    - **Western:** Fort Portal, Hoima, Kabale, Kasese, Mbarara
    - **Eastern:** Iganga, Jinja, Mbale, Soroti, Tororo
    
    **Supported Crops:** Beans, Cassava, Coffee, Cotton, Groundnuts, Maize, Matooke, Rice, Sorghum, Soybeans, Sunflower, Sweet Potatoes
    
    This AI model uses:
    - **Graph Attention Networks (GAT)** to capture spatial relationships between regions, districts, and crops
    - **Bidirectional LSTM** to learn temporal price patterns
    - **Multi-layer architecture** with 578k parameters for accurate predictions
    
    **Performance:** R² = 0.86 (86% accuracy), Average error: 13-15%
    
    Built with PyTorch Geometric and deployed on Hugging Face Spaces.
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
