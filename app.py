import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.load_data import load_data
from src.preprocess import clean_data, engineer_features
from src.analyze_performance import driver_ranking, consistency_analysis, car_performance, rally_specific_analysis
from src.visualize import (set_style, plot_driver_ranking, plot_avg_stage_time_by_driver, 
                           plot_car_comparison, plot_heatmap_performance, 
                           plot_kmeans_clusters, plot_feature_importance, plot_anomalies)
from src.ml_models import perform_kmeans_clustering, train_random_forest, detect_anomalies

# Page config
st.set_page_config(
    page_title="WRC Sports Analytics",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Orbitron:wght@700;900&display=swap');
    
    /* Font styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif;
        color: #E2E8F0 !important;
        text-shadow: 0 0 10px rgba(255, 51, 102, 0.15);
    }
    
    /* App background */
    .stApp {
        background-color: #0A0D1A;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #111529 !important;
        border-right: 1px solid #1E233D;
    }
    
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {
        color: #FF3366 !important;
        font-family: 'Orbitron', sans-serif;
    }
    
    /* Tabs styling */
    button[data-baseweb="tab"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 13px !important;
        font-weight: 700 !important;
        color: #A0AEC0 !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.3s ease;
        padding: 10px 15px !important;
    }
    
    button[data-baseweb="tab"]:hover {
        color: #FF3366 !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FF3366 !important;
        border-bottom: 2px solid #FF3366 !important;
        background-color: rgba(255, 51, 102, 0.05) !important;
    }
    
    /* Metrics panel */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 25px;
        flex-wrap: wrap;
    }
    
    .metric-card {
        flex: 1;
        min-width: 180px;
        background: linear-gradient(135deg, #15192E 0%, #1A203E 100%);
        border: 1px solid #2A2E45;
        border-left: 4px solid #FF3366;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 20px rgba(255, 51, 102, 0.25);
    }
    
    .metric-title {
        font-family: 'Inter', sans-serif;
        font-size: 11px;
        text-transform: uppercase;
        color: #A0AEC0;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 22px;
        font-weight: 900;
        color: #00F0FF;
    }
    
    .metric-sub {
        font-size: 10px;
        color: #718096;
        margin-top: 5px;
    }
    
    /* Banner styling */
    .banner {
        background: linear-gradient(90deg, #FF3366 0%, #15192E 70%, #0A0D1A 100%);
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 30px;
        border-left: 6px solid #00F0FF;
        box-shadow: 0 4px 25px rgba(255, 51, 102, 0.15);
    }
    
    .banner h1 {
        font-family: 'Orbitron', sans-serif;
        margin: 0;
        color: white !important;
        font-size: 32px;
        font-weight: 900;
        letter-spacing: 2px;
    }
    
    .banner p {
        margin: 5px 0 0 0;
        color: #E2E8F0;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data (cached for performance)
@st.cache_data
def load_and_process():
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    return df

df = load_and_process()

# Header Banner
st.markdown("""
<div class="banner">
    <h1>🏎️ WRC RALLY SPORTS ANALYTICS</h1>
    <p>Premium driver telemetry, telemetry consistency metrics, and AI/ML model performance predictions.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("🕹️ Controls")
years = sorted(df['year'].unique(), reverse=True)
selected_year = st.sidebar.selectbox("Select Season Year", years)
rally_list = df[df['year'] == selected_year]['rally'].unique()
selected_rally = st.sidebar.selectbox("Select Rally Event", rally_list)

# Filter dataframe
filtered_df = df[(df['year'] == selected_year) & (df['rally'] == selected_rally)]

# KPIs Summary Metrics
total_stages = filtered_df['stage_id'].nunique()
total_drivers = filtered_df['driver'].nunique()
avg_stage_length = round(filtered_df['stage_length_km'].mean(), 1) if 'stage_length_km' in filtered_df.columns else 0
fastest_speed = round(filtered_df['avg_speed_kmh'].max(), 1) if 'avg_speed_kmh' in filtered_df.columns else 0

st.markdown(f"""
<div class="metric-container">
    <div class="metric-card">
        <div class="metric-title">Active Rally</div>
        <div class="metric-value" style="color: #FF3366;">{selected_rally}</div>
        <div class="metric-sub">Season Year: {selected_year}</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Stages Completed</div>
        <div class="metric-value">{total_stages}</div>
        <div class="metric-sub">Total Rally Stages</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Active Drivers</div>
        <div class="metric-value">{total_drivers}</div>
        <div class="metric-sub">Registered Teams</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Avg Stage Length</div>
        <div class="metric-value" style="color: #00F0FF;">{avg_stage_length} km</div>
        <div class="metric-sub">Average Stage Distance</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Top Speed Recorded</div>
        <div class="metric-value" style="color: #00F0FF;">{fastest_speed} km/h</div>
        <div class="metric-sub">Peak Stage Velocity</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main area tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 Driver Ranking", 
    "🎯 Consistency", 
    "🚗 Car Performance", 
    "📈 Stage Progression",
    "🔥 Performance Heatmap",
    "🤖 AI/ML Insights"
])

with tab1:
    st.subheader("🏆 Driver Standings (Composite Score)")
    st.markdown("Lower composite score indicates better performance (ranks across position, speed, and stage wins).")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        ranking = driver_ranking(filtered_df)
        st.dataframe(
            ranking.style.background_gradient(cmap="flare_r", subset=['composite_score'])
            .highlight_min(subset=['composite_score'], color='#00F0FF'),
            use_container_width=True
        )
    with col2:
        fig_ranking = plot_driver_ranking(ranking)
        st.pyplot(fig_ranking)

with tab2:
    st.subheader("🎯 Telemetry Consistency (Lower CV = More Consistent)")
    st.markdown("Analysis of speed and stage times variability across stages.")
    
    consistency = consistency_analysis(filtered_df)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(consistency, use_container_width=True)
    with col2:
        chart_data = consistency.reset_index().sort_values('cv_stage_time')
        fig, ax = plt.subplots(figsize=(10, 6))

        if (chart_data['stages_completed'] < 2).all():
            st.warning(
                "This rally only has one recorded stage per driver, so a consistency metric "
                "cannot be computed yet. Showing the available stage times instead."
            )
            fallback_data = chart_data.sort_values('mean_stage_time_sec')
            colors = sns.color_palette("flare", len(fallback_data))
            ax.barh(fallback_data['driver'], fallback_data['mean_stage_time_sec'], color=colors)
            ax.set_title("Single Recorded Stage Time by Driver")
            ax.set_xlabel("Stage Time (seconds)")
        else:
            if (chart_data['stages_completed'] < 2).any():
                st.info(
                    "Drivers with fewer than two recorded stages are excluded from the CV chart."
                )
                chart_data = chart_data[chart_data['stages_completed'] >= 2]

            colors = sns.color_palette("crest_r", len(chart_data))
            ax.barh(chart_data['driver'], chart_data['cv_stage_time'], color=colors)
            ax.set_title("Coefficient of Variation by Driver")
            ax.set_xlabel("CV of Stage Time")

        st.pyplot(fig)

with tab3:
    st.subheader("🚗 Car Brand / Model Performance")
    st.markdown("Telemetry aggregates grouped by car manufacturers.")
    
    car_stats = car_performance(filtered_df)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(car_stats, use_container_width=True)
    with col2:
        fig_car = plot_car_comparison(car_stats)
        st.pyplot(fig_car)

with tab4:
    st.subheader("📈 Driver Stage-by-Stage Progression")
    st.markdown("Trace how stage position changes over subsequent stages of the rally.")
    
    drivers = filtered_df['driver'].unique()
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_driver = st.selectbox("Highlight Driver Line", drivers)
        st.info("The highlighted driver's progress line will be displayed in bold neon coral red.")
    with col2:
        stage_view = filtered_df.sort_values(['stage_id', 'driver']).copy()
        stage_count = stage_view['stage_id'].nunique()
        fig, ax = plt.subplots(figsize=(10, 6))

        if stage_count < 2:
            st.warning(
                "This rally selection only has one recorded stage. Showing single stage snapshot."
            )
            single_stage = stage_view.sort_values('stage_position')
            bar_colors = ['#FF3366' if driver == selected_driver else '#00F0FF' for driver in single_stage['driver']]
            ax.barh(single_stage['driver'], single_stage['stage_position'], color=bar_colors)
            ax.invert_yaxis()
            ax.set_xlabel("Stage Position (1 = best)")
            ax.set_ylabel("Driver")
            stage_label = single_stage['stage_name'].iloc[0]
            ax.set_title(f"Stage Position Snapshot – {selected_rally} {selected_year} ({stage_label})")
        else:
            for driver in drivers:
                driver_data = stage_view[stage_view['driver'] == driver]
                ax.plot(
                    driver_data['stage_id'],
                    driver_data['stage_position'],
                    marker='o',
                    label=driver,
                    linewidth=3.5 if driver == selected_driver else 1.2,
                    alpha=1.0 if driver == selected_driver else 0.35,
                    color='#FF3366' if driver == selected_driver else None
                )
            ax.invert_yaxis()
            ax.set_xlabel("Stage Number")
            ax.set_ylabel("Stage Position (1 = best)")
            ax.set_title(f"Stage Position Progression – {selected_rally} {selected_year}")
            ax.set_xticks(sorted(stage_view['stage_id'].unique()))
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        st.pyplot(fig)

with tab5:
    st.subheader("🔥 Performance Heatmap: Driver vs Rally")
    st.markdown("Average performance scores across rallies (lower is better, green is optimal).")
    fig_heatmap = plot_heatmap_performance(df[df['year'] == selected_year])
    st.pyplot(fig_heatmap)

with tab6:
    st.subheader("🤖 AI/ML Insights & Telemetry Models")
    
    # 1. Clustering
    st.markdown("### 1. Driver Archetypes Clustering (K-Means)")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("Groups drivers based on average speed and consistency across the entire database to classify driver profiles.")
        n_clusters = st.slider("Select number of clusters (k)", min_value=2, max_value=5, value=3)
        clusters_df = perform_kmeans_clustering(df, n_clusters=n_clusters)
        if not clusters_df.empty and 'cluster' in clusters_df.columns:
            st.dataframe(
                clusters_df[['driver', 'stages_completed', 'mean_stage_time_sec', 'cv_stage_time', 'cluster']]
                .style.background_gradient(cmap="viridis", subset=['cluster']),
                use_container_width=True
            )
    with col2:
        if not clusters_df.empty and 'cluster' in clusters_df.columns:
            fig_kmeans = plot_kmeans_clusters(clusters_df)
            st.pyplot(fig_kmeans)
        else:
            st.warning("Not enough data to perform clustering.")
            
    st.divider()
    
    # 2. RF feature importance
    st.markdown("### 2. Stage Time Drivers (Random Forest Feature Importance)")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("A Random Forest regression model is trained on the entire telemetry log to predict stage completion times. The feature importance graph highlights which factor matters most.")
        rf_model, importances, rf_data = train_random_forest(df)
        if importances is not None and not importances.empty:
            st.dataframe(importances, use_container_width=True)
    with col2:
        if importances is not None and not importances.empty:
            fig_rf = plot_feature_importance(importances)
            st.pyplot(fig_rf)
        else:
            st.warning("Not enough data to train the Random Forest.")
            
    st.divider()
    
    # 3. Anomaly detection
    st.markdown("### 3. Performance Outliers (Isolation Forest Anomaly Detection)")
    st.write("Identifies statistically anomalous stage times (indicating potential accidents, mechanical failures, or outstanding runs).")
    anomalies_df = detect_anomalies(df)
    if not anomalies_df.empty and 'anomaly' in anomalies_df.columns:
        fig_anomalies = plot_anomalies(anomalies_df)
        st.pyplot(fig_anomalies)
        
        st.markdown("**🚨 Detected Anomalies / Outliers Details (anomaly = -1):**")
        st.dataframe(
            anomalies_df[anomalies_df['anomaly'] == -1][['year', 'rally', 'stage_name', 'driver', 'car', 'stage_length_km', 'stage_time_sec', 'avg_speed_kmh']],
            use_container_width=True
        )
    else:
        st.warning("Not enough data for anomaly detection.")

# Raw data expander
with st.expander("📂 View Raw Telemetry Data (Filtered Selection)"):
    st.dataframe(filtered_df, use_container_width=True)
