# Render Deployment Guide - 100% Free, No Credit Card Required!

## Why Render?

- ✅ **Completely FREE** - No credit card needed
- ✅ **750 hours/month free** (enough for 24/7)
- ✅ **Auto-deploy from GitHub**
- ✅ **Free SSL certificates**
- ✅ **Better for ML models** than most free platforms
- ✅ **No sleep after inactivity** on paid plans (free tier sleeps after 15 min)

## Prerequisites

- GitHub account with code pushed
- Render account (sign up at https://render.com)

## Step-by-Step Deployment

### 1. Sign Up for Render

1. Go to: https://render.com
2. Click **"Get Started"**
3. Sign up with **GitHub** (easiest)
4. Authorize Render to access your repositories

### 2. Create New Web Service

1. Click **"New +"** (top right)
2. Select **"Web Service"**
3. Connect your repository: `otimongjonathan/newImproved`
4. Click **"Connect"**

### 3. Configure Service

Render will auto-detect settings, but verify:

- **Name**: `agri-price-api` (or your choice)
- **Region**: Oregon (or closest to you)
- **Branch**: `main`
- **Runtime**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1`
- **Plan**: **Free**

### 4. Environment Variables (Optional)

Click **"Add Environment Variable"**:
- `FLASK_ENV` = `production`
- `PYTHON_VERSION` = `3.11.6`

### 5. Deploy!

1. Click **"Create Web Service"**
2. Render will start building (5-10 minutes)
3. Watch the build logs in real-time
4. Once complete, you'll get a URL like: `https://agri-price-api.onrender.com`

## Test Your API

```bash
# Health check
curl https://agri-price-api.onrender.com/health

# Make prediction
curl -X POST https://agri-price-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "region": "Central",
    "district": "Kampala",
    "crop": "Maize",
    "forecast_months": 6
  }'
```

## Automatic Deployments

Every time you push to GitHub:
```bash
git add .
git commit -m "Update model"
git push origin main
```

Render automatically redeploys! 🚀

## Free Tier Limitations

- **750 hours/month** (perfect for one app)
- **Sleeps after 15 min inactivity** (wakes up on request in ~30 seconds)
- **512 MB RAM** (enough for your model)
- **Slow build times** (5-10 minutes)

## Upgrade Options (Optional)

If you need:
- **No sleep**: Upgrade to Starter ($7/month)
- **More RAM**: Scale up as needed
- **Faster builds**: Paid plans build faster

## Troubleshooting

### Build Fails

Check logs in Render dashboard. Common issues:

**Missing dependencies:**
```bash
# Update requirements.txt
git add requirements.txt
git commit -m "Update dependencies"
git push
```

**Out of memory:**
Reduce workers in start command:
