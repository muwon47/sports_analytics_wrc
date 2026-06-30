# 🏎️ WRC Rally Sports Analytics Dashboard

A premium, dark-themed telemetry analytics web application and machine learning pipeline built to process and visualize historical sports data from the **World Rally Championship (WRC)** over the past few years. 

This project analyzes historical driver stage times, computes statistical consistency, compares manufacturer performance, and leverages machine learning models to cluster driver archetypes, calculate telemetry feature importances, and identify performance anomalies.

---

## 📊 Core Features & Historical Analysis

The application is structured as a multi-dimensional Streamlit dashboard with a professional telemetry design system, performing deep analysis on historical rally data:

### 🏆 1. Driver Standings & Leaderboards
* Computes composite rankings based on stage finish positions, average speed, and stage wins.
* Renders interactive visual bar charts showing the fastest and most efficient drivers.

### 🎯 2. Telemetry Consistency Analysis
* Tracks the variability in driver stage times across the season.
* Computes the **Coefficient of Variation (CV)** for stage times—lower scores represent highly consistent, stable drivers.

### 🚗 3. Manufacturer Comparison
* Aggregates driver statistics by car brand (Toyota, Hyundai, Ford, etc.).
* Highlights which brand is achieving the highest top speeds and most consistent stage finishes.

### 📈 4. Stage-by-Stage Progression Line Chart
* Renders the path of each driver through the subsequent stages of a selected rally.
* Highlight controls allow users to isolate individual driver lines in a bold, neon coral layout.

### 🔥 5. Rally Performance Heatmap
* Maps average driver performance scores across different rallies and seasons using automated colour mapping.

### 🤖 6. AI & Machine Learning Insights
* **Driver Archetypes Clustering (K-Means)**: Segments drivers into profiles based on telemetry speed, stage completions, and consistency.
* **Telemetry Feature Importance (Random Forest)**: Trains a regression model to determine which attributes (like stage lengths or speeds) have the greatest impact on stage completion times.
* **Outlier & Failure Detection (Isolation Forest)**: Flags anomalous stage times, indicating crashes, mechanical failures, or outstanding outlier drives.

---

## 📂 Project Structure

```
sports_analytics_wrc/
├── .streamlit/             # Streamlit configuration directory
│   └── config.toml         # Theme settings (dark mode, secondary colors)
├── data/                   # Data storage directory
│   └── wrc_data.csv        # Core WRC telemetry dataset
├── src/                    # Modular source code
│   ├── __init__.py         # Package indicator
│   ├── load_data.py        # Dataset ingestion functions
│   ├── preprocess.py       # Data cleaning and feature engineering
│   ├── analyze_performance.py # Math & statistics functions (rankings, CV, car comparisons)
│   ├── visualize.py        # Matplotlib/Seaborn custom plotting functions
│   └── ml_models.py        # K-Means, Random Forest, and Isolation Forest training
├── app.py                  # Streamlit Web Application entrypoint
├── main.py                 # CLI Execution tool for stdout reports
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## 🛠️ Technology Stack

* **Front-end / Dashboard**: [Streamlit](https://streamlit.io/)
* **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
* **Plotting & Visuals**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
* **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/)

---

## 💻 Quick Start & Installation

Ensure you have Python 3.8+ installed. Follow these steps to run the project locally:

### 1. Set Up Virtual Environment
Initialize a clean environment to install dependencies:
```bash
# Initialize venv
python -m venv venv

# Activate venv (Windows)
venv\Scripts\activate

# Activate venv (macOS / Linux)
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Command Line Interface (CLI)
To run a quick terminal-based report of the dataset stats and print ML cluster results:
```bash
python main.py
```

### 4. Start the Web Dashboard
Launch the interactive local Streamlit server:
```bash
streamlit run app.py
```
The dashboard will launch automatically in your browser at `http://localhost:8501`.

---

## 📝 Ignoring Unwanted Files

This project includes a pre-configured `.gitignore` file that prevents local development artifacts from being pushed to GitHub, including:
* Virtual environments (`venv/`)
* Python bytecode (`__pycache__/`, `*.pyc`)
* Streamlit local secrets (`secrets.toml`)
* Local IDE directories (`.vscode/`, `.idea/`)
* Word generation temp files (`~$*`)
