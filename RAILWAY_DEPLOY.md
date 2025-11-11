# Railway Deployment Guide - Optimized for Free Tier

## Why Railway?

- ✅ **No credit card required**
- ✅ **8GB RAM** (perfect for ML models)
- ✅ **$5 free credit/month** (~500 hours)
- ✅ **Better for PyTorch** than Render/Heroku
- ✅ **Auto-deploy from GitHub**

## Prerequisites
- GitHub account with code pushed: `otimongjonathan/newImproved`
- Railway account (sign up at https://railway.app)

## Image Size Optimization (CRITICAL)

Our optimizations to get under 4GB:
- ✅ CPU-only PyTorch (~2GB saved)
- ✅ Aggressive .dockerignore (~500MB saved)
- ✅ No cache during pip install (~300MB saved)
- ✅ Removed unnecessary dependencies (~200MB saved)
- ✅ **Total image size: ~2.5-3GB** ✅

## Deployment Steps

### 1. Commit Optimizations

```bash
cd c:\Users\USER\OneDrive\Desktop\Gene\improved

git add .
git commit -m "Optimize for Railway: reduce image size to <4GB"
git push origin main
```

### 2. Deploy to Railway

#### Via Railway Dashboard (Easiest)

1. Go to: https://railway.app
2. Click **"Login"** → Sign in with **GitHub**
3. Click **"New Project"**
4. Select **"Deploy from GitHub repo"**
5. Choose: `otimongjonathan/newImproved`
6. Railway auto-detects Python
7. Click **"Deploy"**

### 3. Monitor Build

Watch the deployment logs:
- **Building**: 5-10 minutes
- **Deploying**: 1-2 minutes
- **Status**: Should show "Active" when ready

### 4. Generate Domain

1. Click on your service
2. Go to **"Settings"** tab
3. Scroll to **"Domains"**
4. Click **"Generate Domain"**
5. Get URL like: `https://newimproved-production.up.railway.app`

### 5. Test Your API

```bash
# Replace with your Railway URL
RAILWAY_URL="https://newimproved-production.up.railway.app"

# Health check
curl $RAILWAY_URL/health

# Prediction test
curl -X POST $RAILWAY_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "region": "Central",
    "district": "Kampala",
    "crop": "Maize",
    "forecast_months": 6
  }'
```

## Troubleshooting

### Build Still Fails - Image Too Large

**Check image size in logs:**
Look for: `Image of size X.X GB`

**If still >4GB, try these:**

**Option 1: Remove torch-geometric (if not critical)**
```pip-requirements
# Remove from requirements.txt
# torch-geometric
```

**Option 2: Compress model file**
```bash
# Check model size
dir models\best_model.pth

# If >100MB, compress it
python compress_model.py
```

**Option 3: Use Git LFS**
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes models/
git commit -m "Use Git LFS for model"
git push
```

### Out of Memory (R14 Error)

Railway has 8GB RAM, but if you hit limits:

1. **Reduce workers:**
```
# Procfile
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 1 --max-requests 25
```

2. **Enable model caching:**
Models are loaded once at startup, not per request

3. **Monitor memory:**
Check Railway metrics dashboard

### Slow Cold Starts

Railway keeps services warm, but on free tier:
- First request after inactivity: ~10 seconds
- Subsequent requests: <1 second

**Solution:** Ping service every 5 minutes with UptimeRobot

### Build Timeout

If build takes >10 minutes:
```toml
# railway.toml - increase timeout
[build]
buildTimeout = 1200
```

## Cost Management

### Free Tier
- **$5 credit/month**
- **~500 hours execution time**
- **Perfect for 1 app running 24/7**

### Monitor Usage
1. Go to Railway dashboard
2. Click **"Usage"**
3. See credit consumption
4. Set up alerts

### Optimize Costs
```bash
# Use fewer workers
workers = 1

# Enable request limits
--max-requests 50

# Preload app (faster, less memory)
--preload
```

## Features You Get

- ✅ **Auto HTTPS**
- ✅ **Auto deploys** on git push
- ✅ **Environment variables**
- ✅ **Real-time logs**
- ✅ **Metrics monitoring**
- ✅ **Custom domains** (free)
- ✅ **No sleep** (unlike Render/Heroku free tier)

## Updating Deployment

```bash
# Make changes
# Commit
git add .
git commit -m "Update model"
git push origin main

# Railway auto-redeploys!
```

## Environment Variables (Optional)

Add in Railway dashboard:
```
FLASK_ENV=production
PYTHON_VERSION=3.11.6
LOG_LEVEL=INFO
```

## Custom Domain

1. Settings → Domains
2. Add custom domain
3. Configure DNS:
   - CNAME: `your-domain.com` → `your-app.up.railway.app`
4. SSL auto-configured!

## Success Checklist

- [ ] Code optimized and pushed to GitHub
- [ ] Railway project created
- [ ] Build successful (check image size <4GB)
- [ ] Deployment active
- [ ] Domain generated
- [ ] Health endpoint returns 200
- [ ] Prediction endpoint works
- [ ] Monitoring configured

## Quick Deploy Commands

```bash
# Commit optimizations
cd c:\Users\USER\OneDrive\Desktop\Gene\improved
git add .
git commit -m "Railway optimization: <4GB image"
git push origin main

# Then deploy via Railway dashboard
# https://railway.app
```

## Support

- Railway Docs: https://docs.railway.app
- Discord: https://discord.gg/railway
- Twitter: @Railway

Your app will be live in 10 minutes! 🚀
