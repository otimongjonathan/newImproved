# Complete Git Setup Guide - First Time User

## Step 1: Install Git

### Windows
1. Download Git from: https://git-scm.com/download/win
2. Run the installer
3. Use default settings (click "Next" through all options)
4. Verify installation:
```bash
git --version
```

### Already Installed?
Check if Git is already installed:
```bash
git --version
```

## Step 2: Configure Git with Your School Account

Open Git Bash (or Command Prompt/PowerShell) and run:

```bash
# Set your name
git config --global user.name "Your Name"

# Set your school email
git config --global user.email "your-school-email@example.com"

# Verify configuration
git config --list
```

## Step 3: Set Up GitHub Authentication

### Option A: Personal Access Token (Recommended)

1. Go to GitHub: https://github.com
2. Sign in with your school account
3. Click your profile picture (top right) → **Settings**
4. Scroll down → Click **Developer settings** (bottom left)
5. Click **Personal access tokens** → **Tokens (classic)**
6. Click **Generate new token** → **Generate new token (classic)**
7. Give it a name: `Railway Deployment Token`
8. Set expiration: `90 days` or `No expiration`
9. Select scopes:
   - ✅ `repo` (all checkboxes under repo)
   - ✅ `workflow`
10. Click **Generate token**
11. **COPY THE TOKEN NOW** (you won't see it again!)
12. Save it somewhere safe (Notepad, password manager)

### Option B: GitHub CLI (Alternative)

```bash
# Install GitHub CLI from: https://cli.github.com/
# Then authenticate
gh auth login
```

## Step 4: Initialize Your Project

Open terminal in your project folder:

```bash
# Navigate to your project
cd c:\Users\USER\OneDrive\Desktop\Gene\improved

# Initialize Git repository
git init

# Check status
git status
```

## Step 5: Connect to GitHub Repository

```bash
# Add remote repository
git remote add origin https://github.com/Jonah-Ryt/newImproved.git

# Verify remote
git remote -v
```

## Step 6: Stage and Commit Your Files

```bash
# Stage all files
git add .

# Check what's staged
git status

# Commit with message
git commit -m "Initial commit: Agricultural price prediction API for Railway deployment"

# Verify commit
git log --oneline
```

## Step 7: Push to GitHub

### First Push (Using Token)

```bash
# Push to main branch
git push -u origin main
```

**When prompted for credentials:**
- Username: `Jonah-Ryt` (your GitHub username)
- Password: `paste your Personal Access Token` (NOT your GitHub password)

### Alternative: Use Credential Manager

```bash
# Cache credentials for 1 hour
git config --global credential.helper cache

# Or store permanently (Windows)
git config --global credential.helper wincred
```

## Step 8: Verify on GitHub

1. Go to: https://github.com/Jonah-Ryt/newImproved
2. Refresh the page
3. You should see all your files!

## Common Issues & Solutions

### Issue: "Remote repository not found"
**Solution:**
```bash
# Make sure repository exists on GitHub first
# Create it at: https://github.com/new

# Then add remote
git remote remove origin
git remote add origin https://github.com/Jonah-Ryt/newImproved.git
```

### Issue: "Permission denied"
**Solution:**
- Make sure you're using the Personal Access Token, not your password
- Verify token has `repo` permissions

### Issue: "Branch 'main' doesn't exist"
**Solution:**
```bash
# Rename branch to main
git branch -M main

# Then push
git push -u origin main
```

### Issue: "Large files error"
**Solution:**
```bash
# Install Git LFS for model files
git lfs install
git lfs track "*.pth"
git add .gitattributes
git commit -m "Add LFS tracking"
git push
```

## Quick Reference Commands

### Check Status
```bash
git status
```

### Stage Files
```bash
git add .                 # Stage all files
git add filename.py       # Stage specific file
```

### Commit Changes
```bash
git commit -m "Your message here"
```

### Push to GitHub
```bash
git push
git push origin main      # Explicit branch
```

### Pull Latest Changes
```bash
git pull
```

### View History
```bash
git log
git log --oneline
```

### Undo Changes
```bash
git restore filename.py   # Discard local changes
git reset HEAD~1          # Undo last commit (keep changes)
```

## Complete First-Time Workflow

```bash
# 1. Navigate to project
cd c:\Users\USER\OneDrive\Desktop\Gene\improved

# 2. Initialize Git
git init

# 3. Configure user
git config --global user.name "Your Name"
git config --global user.email "your-email@school.edu"

# 4. Add remote
git remote add origin https://github.com/Jonah-Ryt/newImproved.git

# 5. Stage files
git add .

# 6. Commit
git commit -m "Initial commit: Railway deployment ready"

# 7. Rename branch to main
git branch -M main

# 8. Push
git push -u origin main
```

When prompted:
- Username: `Jonah-Ryt`
- Password: `your_personal_access_token`

## After Successful Push

### Deploy to Railway

1. Go to: https://railway.app
2. Sign in with GitHub
3. Click **New Project**
4. Select **Deploy from GitHub repo**
5. Choose `Jonah-Ryt/newImproved`
6. Wait for deployment (5-10 minutes)
7. Get your URL!

## Future Updates

After initial setup, updating is simple:

```bash
# Make changes to your code

# Stage changes
git add .

# Commit
git commit -m "Description of changes"

# Push
git push
```

Railway will automatically redeploy! 🚀

## Need Help?

- Git Documentation: https://git-scm.com/doc
- GitHub Guides: https://guides.github.com/
- Railway Docs: https://docs.railway.app

## Your Repository

**GitHub:** https://github.com/Jonah-Ryt/newImproved.git

**Railway:** https://railway.app (after deployment)
