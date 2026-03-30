#!/bin/bash
# scripts/deploy.sh
# Deployment script for EcoStream AI (Backend -> Render, Frontend -> Vercel)
# Ensure you are logged into Vercel (`vercel login`) before running.

set -e

echo "🚀 Starting EcoStream Deployment..."

# --- 1. DEPLOY FRONTEND (VERCEL) ---
echo "📦 Deploying Frontend to Vercel..."
cd frontend

# Deploy to production (using the Vercel CLI)
# Add --yes to automatically confirm the project link
npx vercel --prod --yes

cd ..
echo "✅ Frontend deployment triggered."

# --- 2. DEPLOY BACKEND (RENDER) ---
echo "📦 Deploying Backend to Render..."

# Replace this with your actual Render Deploy Hook URL
# Found in: Render Dashboard > Web Service > Settings > Deploy Hook
RENDER_DEPLOY_HOOK_URL="https://api.render.com/deploy/srv-placeholder-id?key=placeholder-key"

if [[ "$RENDER_DEPLOY_HOOK_URL" == *"placeholder"* ]]; then
    echo "⚠️  WARNING: Render Deploy Hook URL is a placeholder."
    echo "⚠️  Please update scripts/deploy.sh with your actual Render deploy hook."
else
    # Trigger the Render deploy webhook
    curl -X POST "$RENDER_DEPLOY_HOOK_URL"
    echo "✅ Backend deployment triggered."
fi

echo "🎉 Deployment process complete!"
