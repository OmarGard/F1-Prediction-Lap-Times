# 🏁 F1 Top-3 Predictor AI (2025 Silverstone GP)

This is a personal AI project to **predict the top 3 finishers of a Formula 1 Grand Prix**, with a focus on the **2025 Silverstone GP**. The model leverages real-world F1 data sources, race dynamics, car and driver performance trends, and machine learning techniques to forecast podium outcomes.

---

## 🔍 Objective

Build a machine learning model (using neural networks and tabular ML) to predict the top 3 finishers of an F1 race, based on the **most recent performance trends** leading into the event.

---

## 🚦 Approach

1. **Data Collection**:
   - Gather real telemetry, timing, and weather data from recent races using `FastF1` and/or `OpenF1 API`.
   - Extract features like average race pace, qualifying performance, tire strategy, degradation, etc.

2. **Feature Engineering**:
   - Construct **driver performance snapshots** using data from the 3–5 races preceding Silverstone 2025.
   - Include car/team trends, compound behavior, weather, and more.

3. **Modeling**:
   - Train a neural network or hybrid model to predict **finish probabilities** for each driver.
   - Use ranking loss functions (e.g., RankNet, NDCG@3) or multi-label classification.

4. **Prediction & Evaluation**:
   - Evaluate the model by comparing predictions with actual race results.
   - Validate using backtests and compare with betting odds or expert predictions.

---

## 📦 Project Structure
```
f1-predictor/
├── data/ # Raw and processed race data
├── notebooks/ # Exploratory analysis and modeling notebooks
├── scripts/ # Python scripts for data collection and processing
├── models/ # Saved model weights and training checkpoints
├── main.py # Project entry point (coming soon)
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

---

### Install dependencies:
```bash
pip install -r requirements.txt
```

---

### 📘 License
MIT License. This project is open for learning, research, and experimentation. Not affiliated with FIA, Formula 1, or any racing team.


