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

# 📸 Project Screenshots

## 🚗 Car Performance

<img width="1838" height="962" alt="car" src="https://github.com/user-attachments/assets/9d4cad4d-518a-4e38-882f-c505afdc5a7c" />

Compares vehicle models based on **average stage time**, **average speed**, **finishing position**, and **time gap to the stage winner**, providing insights into overall manufacturer performance.

---

## 🔥 Performance Heatmap

<img width="1847" height="935" alt="consistency" src="https://github.com/user-attachments/assets/b28975fe-48ad-43fb-955e-c461d005fa65" />

Visualizes driver performance across different rallies using a heatmap, enabling quick comparison of consistency and identifying strengths and weaknesses throughout the championship.

---

# 🤖 AI & Machine Learning Insights

## 📊 Driver Clustering (K-Means)

<img width="1846" height="928" alt="insights" src="https://github.com/user-attachments/assets/e8edeb23-4464-4955-9b67-c49b0b8ab905" />

Clusters drivers based on **average stage time** and **performance consistency**, revealing groups of drivers with similar performance characteristics and driving profiles.

---

## 🌲 Feature Importance (Random Forest)

<img width="1838" height="908" alt="stage" src="https://github.com/user-attachments/assets/6345bc9a-110f-4731-a5fc-dba0e6ca369a" />

Applies a **Random Forest Regression** model to identify the factors that most influence stage completion time, highlighting the relative importance of each feature.

---

## 🚨 Anomaly Detection (Isolation Forest)

<img width="1836" height="975" alt="ai" src="https://github.com/user-attachments/assets/7b11341d-0622-468a-b0d5-8f430cb9073a" />

Uses an **Isolation Forest** algorithm to detect anomalous stage performances that may indicate exceptional runs, mechanical failures, crashes, or other unusual rally events.

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

## 👨‍💻 Development & Contributions

This project was developed as part of a collaborative effort to build a data-driven analytics platform for the World Rally Championship (WRC). My contributions focused on both the data pipeline.

### My Contributions

- Contributed to the collection, organization, and preparation of the WRC dataset used throughout the project.
- Performed data cleaning and preprocessing to improve data quality and consistency.
- Engineered analytical features such as average speed, time difference to stage winner and performance scores.
- Assisted in integrating data processing, analytics, and visualization modules like boxplots, heatmaps, charts, etc.
- Participated in testing, debugging, and refining features to improve usability and reliability.

### Skills Demonstrated

- Data Collection & Dataset Creation
- Data Cleaning & Preprocessing
- Data Visualization
- Exploratory Data Analysis (EDA)
- Python Development
- Collaborative Software Development
- Git & GitHub


## 📝 Ignoring Unwanted Files

This project includes a pre-configured `.gitignore` file that prevents local development artifacts from being pushed to GitHub, including:
* Virtual environments (`venv/`)
* Python bytecode (`__pycache__/`, `*.pyc`)
* Streamlit local secrets (`secrets.toml`)
* Local IDE directories (`.vscode/`, `.idea/`)
* Word generation temp files (`~$*`)



