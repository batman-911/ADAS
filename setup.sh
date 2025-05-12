#!/bin/bash

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Setting up environment variables..."
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Downloading pretrained model..."
#wget http://example.com/model.pth -P models/

echo "Setup complete."
