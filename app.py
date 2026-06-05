import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Mall Customer Hierarchical Clustering",
    page_icon="🛍️",
    layout="wide"
)

# ======================================
# LOAD MODEL & SCALER
# ======================================
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load("hierarchical_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        return None, None

# ======================================
# LOAD DATA
# ======================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Mall_Customers.csv")
        clustered_df = pd.read_csv("clustered_mall_customers.csv")
        return df, clustered_df
    except FileNotFoundError:
        return None, None

model, scaler = load_model_and_scaler()
df, clustered_df = load_data()

# ======================================
# ERROR HANDLING
# ======================================
if model is None or scaler is None or df is None or clustered_df is None:
    st.error("❌ Required files not found. Please ensure all model and CSV files exist.")
    st.stop()

# ======================================
# CLUSTER INFO
# ======================================
CLUSTER_INFO = {
    0: "High Value Customers",
    1: "Potential Target Customers",
    2: "Average Customers",
    3: "Loyal Customers",
    4: "Budget Conscious Customers",
    5: "Low Value Customers"
}

# ======================================
# TITLE
# ======================================
st.title("🛍️ Mall Customer Clustering Prediction")
st.markdown("Predict which **customer segment** a person belongs to using ML")

# ======================================
# INPUT SECTION
# ======================================
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", int(df['Age'].min()), int(df['Age'].max()), 30)
    income = st.slider(
        "Annual Income (k$)",
        int(df['Annual Income (k$)'].min()),
        int(df['Annual Income (k$)'].max()),
        50
    )
    spending = st.slider("Spending Score (1-100)", 1, 100, 50)

with col2:
    st.metric("Total Customers", len(df))
    st.metric("Average Income", f"${df['Annual Income (k$)'].mean():.1f}k")
    st.metric("Average Spending", f"{df['Spending Score (1-100)'].mean():.1f}")

# ======================================
# PREDICTION
# ======================================
if st.button("Predict Cluster"):
    input_df = pd.DataFrame({
        "Age": [age],
        "Annual Income (k$)": [income],
        "Spending Score (1-100)": [spending]
    })

    input_scaled = scaler.transform(input_df)
    cluster = model.predict(input_scaled)[0]

    st.success(f"### Predicted Cluster: {cluster}")
    st.info(f"**Customer Type:** {CLUSTER_INFO[cluster]}")

    # ======================================
    # CLUSTER STATS
    # ======================================
    cluster_data = clustered_df[clustered_df['Cluster'] == cluster]

    st.markdown("### 📊 Cluster Statistics")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Customers in Cluster", len(cluster_data))
    with c2:
        st.metric("Avg Income", f"${cluster_data['Annual Income (k$)'].mean():.1f}k")
    with c3:
        st.metric("Avg Spending", f"{cluster_data['Spending Score (1-100)'].mean():.1f}")

    # ======================================
    # VISUALIZATION
    # ======================================
    st.markdown("### 📈 Cluster Visualization")

    fig = px.scatter(
        clustered_df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color=clustered_df["Cluster"].astype(str),
        title="Customer Segments"
    )

    fig.add_scatter(
        x=[income],
        y=[spending],
        mode="markers",
        marker=dict(size=14, color="red", symbol="diamond"),
        name="Your Input"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================================
    # PIE CHART
    # ======================================
    st.markdown("### 🥧 Cluster Distribution")

    cluster_counts = clustered_df['Cluster'].value_counts().sort_index()

    fig_pie = go.Figure(data=[go.Pie(
        labels=[f"Cluster {i}" for i in cluster_counts.index],
        values=cluster_counts.values,
        hole=0.4
    )])

    st.plotly_chart(fig_pie, use_container_width=True)