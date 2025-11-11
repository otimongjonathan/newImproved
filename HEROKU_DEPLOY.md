# Heroku Deployment Guide - GitHub Student Pack

## Prerequisites
- GitHub Student Pack activated (https://education.github.com/pack)
- Heroku account (sign up at https://heroku.com)
- Git installed
- Heroku CLI installed

## Step 1: Install Heroku CLI

### Windows
Download from: https://devcenter.heroku.com/articles/heroku-cli

Or use npm:
```bash
npm install -g heroku
```

Verify installation:
```bash
heroku --version
```

## Step 2: Login to Heroku

```bash
heroku login
```

This opens a browser window - login with your Heroku account.

## Step 3: Create Heroku App

```bash
# Navigate to your project
cd c:\Users\USER\OneDrive\Desktop\Gene\improved

# Create Heroku app
heroku create agri-price-predictor

# Or let Heroku generate a name
heroku create
```

## Step 4: Add Buildpacks

```bash
# Add Python buildpack
heroku buildpacks:set heroku/python

# Use Heroku-22 stack (better for ML)
heroku stack:set heroku-22
```

## Step 5: Configure Environment

```bash
# Set to production
heroku config:set FLASK_ENV=production

# Optimize for limited memory
heroku config:set WEB_CONCURRENCY=1
heroku config:set PYTHONUNBUFFERED=1
```

## Step 6: Deploy to Heroku

```bash
# Add Heroku remote (if not already added)
heroku git:remote -a agri-price-predictor

# Push to Heroku
git push heroku main
```

**First deployment takes 5-10 minutes.**

## Step 7: Scale Dynos

```bash
# Scale web dyno
heroku ps:scale web=1

# Check status
heroku ps
```

## Step 8: Open Your App

```bash
# Open in browser
heroku open

# Or get the URL
heroku info
```

## Step 9: Test Your API

```bash
# Get your app URL
APP_URL=$(heroku info -s | grep web_url | cut -d= -f2)

# Test health endpoint
curl ${APP_URL}health

# Test prediction
curl -X POST ${APP_URL}predict \
  -H "Content-Type: application/json" \
  -d '{
    "region": "Central",
    "district": "Kampala",
    "crop": "Maize",
    "forecast_months": 6
  }'
```

## Monitor & Debug

### View Logs
```bash
# Real-time logs
heroku logs --tail

# Recent logs
heroku logs --num=100
```

### Check Dyno Status
```bash
heroku ps
```

### Restart App
```bash
heroku restart
```

## GitHub Student Pack Benefits

With your student pack, you get:
- **Eco Dyno credits** ($13/month value)
- Better performance than free tier
- 512 MB RAM (upgradeable)

### Activate Student Benefits
1. Go to: https://heroku.com/github-students
2. Verify GitHub Student Pack
3. Get your credits

## Troubleshooting

### Slug Size Too Large

If you see "Slug size exceeds limit":

```bash
# Add .slugignore file
# Already created above

# Commit and redeploy
git add .slugignore
git commit -m "Add slugignore"
git push heroku main
```

### Memory Issues (R14 errors)

```bash
# Reduce workers in Procfile
web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1 --threads 1 --max-requests 50

# Commit and push
git add Procfile
git commit -m "Reduce workers"
git push heroku main
```

### Model File Too Large

If model >300MB, use Git LFS:

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.pth"

# Commit
git add .gitattributes models/
git commit -m "Use Git LFS for model"

# Add Heroku LFS buildpack
heroku buildpacks:add https://github.com/raxod502/heroku-buildpack-git-lfs

# Push
git push heroku main
```

### Build Timeout

```bash
# Increase timeout
heroku config:set BUILD_TIMEOUT=1200
```

### Application Error

Check logs:
```bash
heroku logs --tail
```

Common fixes:
```bash
# Ensure PORT is set correctly in app.py
# Already configured

# Check Procfile
heroku run cat Procfile
```

## Update Deployment

After making changes:

```bash
# Stage changes
git add .

# Commit
git commit -m "Update application"

# Push to GitHub
git push origin main

# Deploy to Heroku
git push heroku main
```

## Useful Commands

```bash
# Run Python shell on Heroku
heroku run python

# Run bash on Heroku
heroku run bash

# Check config vars
heroku config

# Add config var
heroku config:set KEY=value

# View app info
heroku info

# View releases
heroku releases

# Rollback to previous version
heroku rollback
```

## Cost Management

### Free Tier Limits
- 550-1000 dyno hours/month (with student pack)
- 512 MB RAM per dyno
- Sleeps after 30 mins inactivity

### Prevent Sleep
Use a service like:
- https://uptimerobot.com (free)
- https://cron-job.org (free)

Ping your app every 25 minutes.

## Production Checklist

- [x] Procfile configured
- [x] runtime.txt set
- [x] requirements.txt optimized
- [x] .slugignore added
- [x] Environment variables set
- [ ] SSL enabled (automatic)
- [ ] Custom domain (optional)
- [ ] Monitoring setup
- [ ] Logs configured

## Custom Domain (Optional)

If you have a domain from Student Pack (Namecheap):

```bash
# Add domain
heroku domains:add www.yourdomain.com

# Get DNS target
heroku domains

# Configure DNS at Namecheap:
# CNAME: www -> your-app-name.herokuapp.com
```

## Resources

- Heroku Docs: https://devcenter.heroku.com/
- Student Pack: https://www.heroku.com/github-students
- Support: https://help.heroku.com/

## Quick Deploy Summary

```bash
# One-time setup
heroku login
heroku create agri-price-predictor
heroku buildpacks:set heroku/python
heroku config:set FLASK_ENV=production WEB_CONCURRENCY=1

# Deploy
git push heroku main
heroku ps:scale web=1
heroku open
```

Your app will be live at: `https://agri-price-predictor.herokuapp.com` 🚀
