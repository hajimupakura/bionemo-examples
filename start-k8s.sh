#!/bin/bash

echo "===== Kubernetes Environment Startup Script ====="

# Step 1: Check and start Docker if needed
echo "Step 1: Checking Docker status..."
if ! systemctl is-active --quiet docker; then
    echo "Docker is not running. Starting Docker..."
    sudo systemctl start docker
    sleep 3
    
    if ! systemctl is-active --quiet docker; then
        echo "Failed to start Docker. Please check Docker installation."
        exit 1
    fi
else
    echo "Docker is already running."
fi

# Step 2: Check Minikube status
echo "Step 2: Checking Minikube status..."
MINIKUBE_STATUS=$(minikube status --format={{.Host}} 2>/dev/null || echo "Not Running")

if [ "$MINIKUBE_STATUS" != "Running" ]; then
    echo "Minikube is not running. Starting Minikube..."
    minikube start
    
    if [ $? -ne 0 ]; then
        echo "Failed to start Minikube. Please check Minikube installation."
        exit 1
    fi
else
    echo "Minikube is already running."
fi

# Step 3: Verify kubectl is working
echo "Step 3: Verifying kubectl configuration..."
if ! kubectl cluster-info &>/dev/null; then
    echo "kubectl is not properly configured. Trying to fix..."
    minikube update-context
    
    if ! kubectl cluster-info &>/dev/null; then
        echo "Failed to configure kubectl. Please check kubectl installation."
        exit 1
    fi
else
    echo "kubectl is properly configured."
fi

# Step 4: Enable metrics-server for better dashboard experience
echo "Step 4: Enabling metrics-server addon..."
METRICS_STATUS=$(minikube addons list -o json | grep -A3 "metrics-server" | grep "enabled" | awk -F': ' '{print $2}' | tr -d ',"')

if [ "$METRICS_STATUS" != "true" ]; then
    echo "Enabling metrics-server addon..."
    minikube addons enable metrics-server
else
    echo "Metrics-server addon is already enabled."
fi

# Step 5: Start Kubernetes Dashboard
echo "Step 5: Starting Kubernetes Dashboard..."
echo "Dashboard will open in your browser. Press Ctrl+C when you want to stop the dashboard."
echo "The dashboard will continue to run in the background until you run 'minikube stop'."
echo ""
echo "Starting dashboard now..."
minikube dashboard

echo "===== Kubernetes Environment Setup Complete ====="
