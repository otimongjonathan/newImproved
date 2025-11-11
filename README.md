---
title: Agricultural Input Price Predictor - Uganda
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🌾 Agricultural Input Price Predictor - Uganda

AI-powered tool for predicting agricultural input prices in Uganda using deep learning.

## 🚀 Live Demo

**Try it here:** [Hugging Face Space](https://huggingface.co/spaces/YOUR-USERNAME/agri-price-predictor)

## ✨ Features

- 📊 Predict prices for 5 agricultural inputs (seeds, fertilizer, herbicide, pesticide, labor)
- 🌍 Support for all major regions in Uganda (Central, Eastern, Western, Northern)
- 🌾 Multiple crops: Maize, Coffee, Sweet Potatoes
- 📅 Forecast 1-12 months ahead
- 💰 See total cost estimates for your farm

## 🧠 Technology

- **Model:** Hybrid GAT-RNN (Graph Attention Network + Recurrent Neural Network)
- **Framework:** PyTorch
- **Interface:** Gradio
- **Deployment:** Hugging Face Spaces

## 📖 How to Use

1. Select your **region** and **district**
2. Choose your **crop type**
3. Enter your **farm size** in acres
4. Set the **forecast period** (1-12 months)
5. Click **"Predict Prices"**
6. View detailed price forecasts and total costs!

## 🛠️ Local Development

```bash
# Clone repository
git clone https://github.com/otimongjonathan/newImproved.git
cd newImproved

# Install dependencies
pip install -r requirements.txt

# Run Gradio app
python app.py

# Open http://localhost:7860 in your browser
```

## 📊 Model Details

The model uses:
- Historical price data (2020-2024)
- Seasonal patterns and trends
- Regional economic factors
- Crop-specific characteristics
- Market dynamics

**Accuracy:** The model achieves competitive performance on test data with consideration for:
- Monthly inflation rates (0.8%)
- Supply chain volatility (3%)
- Regional price variations
- Weather-based risk factors
- Seasonal multipliers

## 📁 Project Structure
