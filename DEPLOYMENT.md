# Heroku Deployment Guide

## Prerequisites
- Heroku CLI installed: https://devcenter.heroku.com/articles/heroku-cli
- Git installed
- Heroku account created

## Files Required for Deployment
✅ `requirements.txt` - Python dependencies
✅ `Procfile` - Tells Heroku how to run the app
✅ `runtime.txt` - Python version specification
✅ `wsgi.py` - Production WSGI entry point
✅ `.gitignore` - Excludes unnecessary files
✅ `models/best_model.pth` - Trained model weights
✅ `*.csv` datasets

## Step-by-Step Deployment

### 1. Initialize Git Repository (if not already done)
```bash
cd c:\Users\USER\OneDrive\Desktop\Gene\improved
git init
git add .
git commit -m "Initial commit for Heroku deployment"
```

### 2. Create Heroku App
```bash
heroku login
heroku create agricultural-predictor-ug
```

### 3. Set Environment Variables
```bash
heroku config:set SECRET_KEY="your-production-secret-key-here"
heroku config:set FLASK_ENV=production
```

### 4. Deploy to Heroku
```bash
git push heroku main
```
(If your branch is `master`, use `git push heroku master`)

### 5. Scale the Dyno
```bash
heroku ps:scale web=1
```

### 6. Open Your App
```bash
heroku open
```

### 7. View Logs
```bash
heroku logs --tail
```

## Troubleshooting

### Memory Issues
If you get R14 (Memory quota exceeded) errors:
- Reduce model size
- Use smaller batch sizes
- Upgrade to Hobby dyno ($7/month, 512MB RAM)

```bash
heroku ps:resize web=hobby
```

### Timeout Issues
If requests timeout:
- Increase timeout in Procfile (already set to 120s)
- Optimize model inference speed

### Slug Size Too Large
If deployment fails due to slug size (>500MB):
- Remove unnecessary CSV files
- Use `.slugignore` to exclude training logs
- Compress model files

## Post-Deployment

### Monitor Performance
```bash
heroku logs --tail
heroku ps
```

### Update Application
```bash
git add .
git commit -m "Update description"
git push heroku main
```

### Add Custom Domain (Optional)
```bash
heroku domains:add www.yourapp.com
```

## Cost Optimization
- **Free Dyno**: Good for testing (sleeps after 30 min inactivity)
- **Hobby Dyno** ($7/mo): No sleep, 512MB RAM
- **Standard Dyno** ($25/mo): Better for production

## Alternative: Railway Deployment
Railway offers better free tier and easier setup:
1. Connect GitHub repo
2. Auto-detects Python app
3. Deploys automatically on push

Railway dashboard: https://railway.app
