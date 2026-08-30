#!/usr/bin/env bash
# =============================================================================
# Deployment Script for Streamlit + Cloudflared Tunnel on Google Colab
# =============================================================================

echo "[1/4] Installing Streamlit and Project Requirements..."
pip install -q streamlit scikit-learn pandas numpy matplotlib seaborn xgboost

echo "[2/4] Downloading Cloudflared binary..."
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared

echo "[3/4] Launching Streamlit Server in the background on port 8501..."
streamlit run app/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false &>/content/streamlit_logs.txt &

sleep 3

echo "[4/4] Opening Public Cloudflared Tunnel..."
echo "============================================================================="
echo " Look for the generated URL ending with .trycloudflare.com below to access UI:"
echo "============================================================================="
./cloudflared tunnel --url http://localhost:8501 --no-autoupdate
