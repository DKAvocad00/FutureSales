#!/bin/bash

while true; do
    echo "Choose script to run:"
    echo "1. Train Model"
    echo "2. Predict Model"
    echo "3. exit"
    read choice

    case $choice in
        1) python /app/src/models/train_model.py ;;
        2) python /app/src/models/predict_model.py ;;
        3) echo "Stopping container" && exit ;;
        *) echo "Incorrect choice. Please, select 1, 2 or 3." ;;
    esac

    echo "Chosen script is ended."
    echo "Returning to the script selection..."
done