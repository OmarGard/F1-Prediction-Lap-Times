# 🏁 F1 Top-3 Predictor AI (2025 Silverstone GP)

This is a personal AI project to **predict the top 3 finishers of a Formula 1 Grand Prix**, focused on the **2025 Silverstone GP**. The model uses real F1 data, race dynamics, driver and team performance trends, and machine learning techniques to forecast the podium.

---

## 🔍 Objective

Build a machine learning model to predict the top 3 finishers of the 2025 Silverstone GP, using **recent performance trends** prior to the event.

---

## 🚦 Pipeline and Approach

1. **Data Collection**:
   - Real telemetry, timing, and weather data from recent races is obtained using `FastF1` and `Open Weather API` 
   - Features such as clean air race pace, qualifying times, team score, and track conditions are extracted.

2. **Processing and Feature Engineering**:
   - Driver performance snapshots are generated using qualifying and race data from Silverstone 2024 and 2025.
   - Sector averages, clean air race pace, and weather factors are calculated and integrated.
   - A wet performance factor is added if the forecast requires it.

3. **Modeling**:
   - A machine learning model (e.g., GradientBoostingRegressor) is trained to predict each driver's total race time.
   - Missing values are imputed and weather, team, qualifying, and race pace features are combined.

4. **Prediction and Evaluation**:
   - Race times are predicted and sorted to obtain the podium.
   - The model is evaluated with MAE and feature importance is visualized.

---

## 📦 Project Structure
```
f1-predictor/
├── data/ # Raw and processed race and weather data
├── notebooks/ # Exploratory analysis and modeling notebooks
├── scripts/ # Python scripts for data processing and extraction
│   ├── session_data.py # Session data loading and cleaning
│   ├── racepace.py # Clean air race pace extraction
│   ├── qualifying.py # Qualifying data extraction
│   ├── team_performance.py # Team performance scoring
│   └── utils.py # Feature and matching utilities
├── models/ # Trained model weights and checkpoints
├── main.py # Main prediction and visualization pipeline
└── README.md # Project documentation
```

---

## ⚙️ Setup

### Requirements
- Python 3.12+
- FastF1
- Pandas, NumPy
- Scikit-learn
- XGBoost or LightGBM (optional)
- PyTorch or TensorFlow (for deep models)
- fuzzywuzzy (for fuzzy name matching)

---

### Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Pipeline Execution

The main pipeline is in `main.py` and includes:

- Loading and processing session and weather data.
- Extraction of qualifying, race pace, and team features.
- Integration of weather conditions.
- Model training and evaluation.
- Visualization of results and feature importance analysis.

---

## 📘 License
MIT License. Open project for learning, research, and experimentation. Not affiliated with FIA, Formula 1, or any team.


