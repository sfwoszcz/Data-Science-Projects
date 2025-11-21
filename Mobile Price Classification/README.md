# Mobile Phone Price Classification

This mini-project classifies mobile phones into **price ranges** (`l`=low, `m`=medium, `h`=high)
using features such as **battery power**, **front camera megapixels**, **internal memory**, and **Bluetooth**.

## Dataset
File: `MobilePhone.csv`

Columns (observed):
- `battery_power` (int)
- `blue` (0/1, Bluetooth present)
- `dual_sim` (0/1)
- `frontcamermegapixels` (int)
- `four_g` (0/1)
- `int_memory` (int, GB)
- `three_g` (0/1)
- `touch_screen` (0/1)
- `Price Range` (label: `l`, `m`, `h`)

> If you do not have the CSV, place it at `/mnt/data/MobilePhone.csv`
> or adjust the paths inside the scripts.

## What’s included

- `mobile_visualization.py`  
  Creates the required scatter plots, with points colored by price class, and saves:
  - `plot_battery_vs_memory.png`
  - `plot_frontcam_vs_bluetooth.png`
  - `plot_memory_vs_frontcam.png`
  - `plot_3d_battery_blue_frontcam.png`

- `mobile_classification.py`  
  Trains and evaluates two models **using only** `battery_power` and `frontcamermegapixels`:
  - `KNeighborsClassifier` (with `StandardScaler`)
  - `RandomForestClassifier`
  Prints accuracy, confusion matrix, and classification report.

## Setup

```bash
# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## Run

```bash
# Generate the 2D and 3D scatter plots
python mobile_visualization.py

# Train KNN and RandomForest on the two selected features
python mobile_classification.py
```

## Notes

- Classification restricts features to **Battery Power** and **Front Camera Megapixels**, per the exercise.
- Train/test split uses stratification and a fixed random seed for reproducibility.
- KNN is wrapped in a pipeline with `StandardScaler`; Random Forest does not require scaling.
- If your data lives elsewhere, change the `DATA_PATH` constant in each script.

## Optional: GitHub integration

Suggested structure:
```
mobile-price-classification/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ MobilePhone.csv   # place dataset here or adjust path
├─ mobile_visualization.py
└─ mobile_classification.py
```

Initialize and push:
```bash
git init
git add .
git commit -m "Case Study 2.C.02: visualization + classification (KNN & RF)"
git branch -M main
git remote add origin https://github.com/<your-username>/mobile-price-classification.git
git push -u origin main
```

## License
MIT
