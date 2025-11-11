# Heroku GitHub Student Pack Setup Guide

## Step 1: Activate GitHub Student Pack

1. Visit: https://education.github.com/pack
2. Verify you're a student (if not already verified)
3. Provide school email or upload proof of enrollment

## Step 2: Claim Heroku Credits

1. Go to: https://www.heroku.com/github-students
2. Click **"Get access by connecting your GitHub account"**
3. Sign in to Heroku
4. Authorize GitHub connection
5. Credits will be added to your account

## Step 3: Verify Heroku Account

**Important:** Even with Student Pack, you need to add a payment method for verification.

1. Visit: https://heroku.com/verify
2. Add credit/debit card
3. Complete verification

**Don't worry:** You won't be charged if you stay within free tier + student credits!

## Step 4: Check Your Benefits

Visit: https://dashboard.heroku.com/account/billing

You should see:
- **Eco Dyno credits**: 1000 hours/month (12 months)
- **Value**: $13/month
- **Total**: $156 over 12 months

## Step 5: Deploy Your App

```bash
# Create app
heroku create your-app-name

# Deploy
git push heroku main

# Use Eco dyno (free with credits)
heroku ps:scale web=1:eco
```

## Understanding Heroku Pricing

### Free Tier (No Credits Needed)
- **Eco Dynos**: $5/month OR free with Student Pack credits
- **Sleep after 30 min inactivity**
- **550 hours/month free** (without Student Pack)

### With Student Pack Credits
- **1000 Eco dyno hours/month** (enough for 24/7 uptime)
- **No sleep**
- **Better performance**

## Cost Monitoring

```bash
# Check current usage
heroku ps

# View billing
heroku account

# Monitor logs
heroku logs --tail
```

## What You Get with Student Pack

1. **Heroku Eco Dyno credits** - $156/year value
2. **No sleep on free apps**
3. **More dyno hours**
4. **Priority support access**

## Additional Student Pack Benefits

- **Namecheap**: Free domain name + SSL for 1 year
- **DigitalOcean**: $200 credit
- **Azure**: $100 credit
- **MongoDB Atlas**: $50 credit
- **Stripe**: Waived transaction fees

## Frequently Asked Questions

**Q: Will I be charged?**
A: No, as long as you use Eco dynos and stay within 1000 hours/month.

**Q: What happens after 12 months?**
A: Credits expire. You can:
- Add payment method to continue
- Migrate to another platform
- Apply for Student Pack renewal

**Q: Do I need a credit card?**
A: Yes, for account verification only. Won't be charged with active credits.

**Q: Can I use multiple apps?**
A: Yes! 1000 hours/month can be split across multiple apps.

## Deployment Commands

```bash
# Login
heroku login

# Create app
heroku create agri-price-api

# Set buildpack
heroku buildpacks:set heroku/python

# Configure
heroku config:set FLASK_ENV=production

# Deploy
git push heroku main

# Scale to Eco dyno (uses student credits)
heroku ps:scale web=1:eco

# Open app
heroku open
```

## Support

- Heroku Docs: https://devcenter.heroku.com/
- Student Pack: https://education.github.com/pack
- Heroku Support: https://help.heroku.com/

## Verification Checklist

- [ ] GitHub Student Pack activated
- [ ] Heroku credits claimed
- [ ] Payment method added (verification)
- [ ] Account verified
- [ ] Credits visible in dashboard
- [ ] Ready to deploy!
