#!/bin/bash

# Navigate to project folder
cd /home/adrian/VisionBoard-Proj || exit

mkdir -p logs

# Activate virtual environment
source venv/bin/activate

LOG_FILE="logs/error_$(date +%Y-%m-%d_%H-%M-%S).txt"

# Run the main Python app
python3 main.py > "$LOG_FILE" 2>&1
