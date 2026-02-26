# GitHub Deployment Guide

## Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and log in to your account
2. Click the **"+"** icon in the top right → **New repository**
3. Enter repository name: `aqi-tamil-nadu-analysis`
4. Choose **Public** (to enable GitHub Pages)
5. Do NOT initialize with README (we already have one)
6. Click **Create repository**

## Step 2: Push to GitHub

The project is already committed locally. Now add your GitHub repository as the remote and push:

```bash
cd "d:\vs\anti\GroceryStoreDataset"

# Add your GitHub repository URL (replace USERNAME with your GitHub username)
git remote add origin https://github.com/USERNAME/aqi-tamil-nadu-analysis.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 3: Enable GitHub Pages (Optional)

To deploy the documentation on GitHub Pages:

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under "Source", select **Deploy from a branch**
4. Select **main** branch and **/root** folder
5. Click **Save**

Your README will be accessible at: `https://USERNAME.github.io/aqi-tamil-nadu-analysis/`

## Step 4: Deploy Streamlit Apps on Streamlit Cloud (Recommended)

For the interactive dashboard, deploy on Streamlit Cloud:

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with GitHub account
3. Click **New app**
4. Select your repository: `aqi-tamil-nadu-analysis`
5. Set:
   - Main file path: `dashboard/app.py`
   - Python version: 3.9+
6. Click **Deploy**

Your live dashboard will be available at: `https://share.streamlit.io/USERNAME/aqi-tamil-nadu-analysis/main/dashboard/app.py`

## Step 5: Monitor Deployment

After pushing:

1. Check GitHub Actions tab for any automated workflows
2. Verify all files are pushed: `git status` should show "nothing to commit"
3. Monitor Streamlit deployment logs if using Streamlit Cloud

## Current Repository Status

```
Repository: aqi-tamil-nadu-analysis
Branch: main
Local Commit: 44aba18
Status: Ready to push
Files: 47 committed
```

## Commands Summary

```bash
# View current configuration
git remote -v

# Push updates in future
git push origin main

# Check deployment status
git log --oneline
```

## Troubleshooting

**Authentication Error?**
- Use Personal Access Token (PAT) instead of password
- Settings → Developer settings → Personal access tokens → Generate new token

**Files not pushing?**
- Check `.gitignore` for excluded files
- Run: `git add .` && `git commit -m "message"` && `git push`

**Want to update README on GitHub?**
- Edit locally and push: `git add README.md && git commit -m "Update README" && git push`
