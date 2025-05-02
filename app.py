import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os

# -------------- Theme Setup --------------
st.set_page_config(
    page_title="Finance ML Pro App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations and pastel theme
st.markdown("""
    <style>
    /* Main Theme Colors */
    :root {
        --primary-color: #a8d5e2;
        --secondary-color: #e3f2fd;
        --accent-color: #42a5f5;
        --text-color: black;
        --success-color: #81c784;
        --warning-color: #ffb74d;
        --error-color: #e57373;
    }

    /* Main App Styling */
    body {
        background-color: var(--secondary-color);
        font-family: 'Segoe UI', sans-serif;
        color: var(--text-color);
    }

    .main, .stApp {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    /* Header Styling */
    .stMarkdown h1 {
        color: black;
        font-size: 2.5em;
        margin-bottom: 20px;
        animation: fadeInDown 1s ease;
    }

    .stMarkdown h2 {
        color: black;
        font-size: 2em;
        margin-bottom: 15px;
        animation: fadeInLeft 1s ease;
    }

    .stMarkdown h3 {
        color: black;
        font-size: 1.5em;
        margin-bottom: 10px;
        animation: fadeInRight 1s ease;
    }

    /* Button Styling */
    .stButton>button {
        background-color: var(--accent-color);
        color: black;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
        animation: fadeInUp 1s ease;
    }

    .stButton>button:hover {
        background-color: #1e88e5;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Metric Cards */
    .stMetric {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        animation: fadeIn 1s ease;
    }

    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--primary-color) !important;
        padding: 20px;
        border-radius: 0 15px 15px 0;
    }

    [data-testid="stSidebar"] * {
        color: black !important;
    }

    /* Success Messages */
    .stSuccess {
        background-color: var(--success-color);
        color: black;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        animation: fadeIn 0.5s ease;
    }

    /* Warning Messages */
    .stWarning {
        background-color: var(--warning-color);
        color: black;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        animation: fadeIn 0.5s ease;
    }

    /* Error Messages */
    .stError {
        background-color: var(--error-color);
        color: black;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        animation: fadeIn 0.5s ease;
    }

    /* Text Elements */
    .stTextInput > label,
    .stSelectbox > label,
    .stRadio > label,
    .stMultiSelect > label,
    .stMetricLabel,
    .stMetricValue,
    .stDataFrame,
    .stMarkdown,
    .css-1v3fvcr,
    .css-qrbaxs,
    .css-10trblm,
    .css-1d391kg,
    label, span, p, div, th, td {
        color: black !important;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes fadeInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        animation: fadeIn 1s ease;
    }

    /* Plot Styling */
    .element-container {
        animation: fadeIn 1s ease;
    }

    /* Input Fields */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 2px solid var(--primary-color);
        padding: 8px;
        transition: all 0.3s ease;
    }

    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 5px rgba(66, 165, 245, 0.5);
    }

    /* Progress Bar */
    .stProgress > div > div > div {
        background-color: var(--accent-color);
        border-radius: 10px;
    }

    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: var(--accent-color);
        color: black;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    </style>
""", unsafe_allow_html=True)

# -------------- Session State --------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

# -------------- Sidebar Navigation --------------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📂 Load Data", "🧹 Preprocessing", 
                                 "🔢 Feature Selection", "🧠 Train Model", "📈 Results"])

# -------------- Helper Functions --------------
def convert_to_numeric(value):
    """Converts values like '412.60K' to numeric format."""
    if isinstance(value, str):
        if 'K' in value:
            return float(value.replace('K', '')) * 1e3
        elif 'M' in value:
            return float(value.replace('M', '')) * 1e6
    return value

def save_model(model, model_name):
    """Save the trained model to disk."""
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, f'models/{model_name}.joblib')

def load_model(model_name):
    """Load a saved model from disk."""
    try:
        return joblib.load(f'models/{model_name}.joblib')
    except:
        return None

# -------------- 🏠 Home --------------
if page == "🏠 Home":
    st.title("AF3005: Financial ML Application")
    st.image("https://media.giphy.com/media/LHZyixOnHwDDy/giphy.gif", width=500)
    st.markdown("""
    ### 🌟 Welcome to the Enhanced ML App!
    - Upload or fetch financial data
    - Clean, preprocess, and select features
    - Choose from **multiple ML models**
    - Train, evaluate, and visualize predictions

    **Features**:
    - Multiple ML models (Linear Regression, Random Forest, SVR, KNN, Decision Tree)
    - Model comparison and evaluation
    - Interactive visualizations
    - Data preprocessing and feature selection
    - Model saving and loading
    - Beautiful UI with dark/light theme
    """)

# -------------- 📂 Load Data --------------
elif page == "📂 Load Data":
    st.title("📂 Load Financial Data")
    source = st.radio("Select Data Source", ["Upload CSV", "Fetch from Yahoo Finance"])

    if source == "Upload CSV":
        file = st.file_uploader("Upload your CSV", type=["csv"])
        if file:
            try:
                df = pd.read_csv(file)
                df = df.map(convert_to_numeric)
                st.session_state.df = df.dropna()
                st.write("Data Preview:")
                st.dataframe(df.head())
                st.success("Data loaded and cleaned successfully!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    else:
        ticker = st.text_input("Enter Ticker (e.g., AAPL)")
        if st.button("Fetch"):
            try:
                df = yf.download(ticker, period="1y")
                if not df.empty:
                    df = df.map(convert_to_numeric)
                    st.session_state.df = df.dropna()
                    st.write("Data Preview:")
                    st.dataframe(df.tail())
                    st.success(f"Loaded 1Y data for {ticker}")
                else:
                    st.error("Invalid ticker or no data available")
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")

# -------------- 🧹 Preprocessing --------------
elif page == "🧹 Preprocessing":
    st.title("🧹 Data Cleaning")
    if st.session_state.df is None:
        st.warning("Please load data first.")
    else:
        df = st.session_state.df.copy()
        
        # Show data info
        st.subheader("Data Information")
        st.write(f"Shape: {df.shape}")
        st.write("Missing values:")
        st.write(df.isnull().sum())
        
        # Data cleaning options
        st.subheader("Cleaning Options")
        if st.checkbox("Remove rows with missing values"):
            df = df.dropna()
        
        if st.checkbox("Remove duplicate rows"):
            df = df.drop_duplicates()
        
        # Feature scaling
        st.subheader("Feature Scaling")
        scale_features = st.multiselect("Select features to scale", df.columns)
        if scale_features:
            scaler = StandardScaler()
            df[scale_features] = scaler.fit_transform(df[scale_features])
            st.session_state.scaler = scaler
        
        st.session_state.df = df
        st.success("Data preprocessing complete!")
        st.write("Processed Data Preview:")
        st.dataframe(df.head())

# -------------- 🔢 Feature Selection --------------
elif page == "🔢 Feature Selection":
    st.title("🔢 Select Features")
    df = st.session_state.df
    if df is not None:
        target = st.selectbox("Target (Y)", df.columns)
        features = st.multiselect("Features (X)", [c for c in df.columns if c != target])

        if st.button("Split Data"):
            try:
                X = df[features]
                y = df[target]
                
                # Add test size slider
                test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                st.session_state.update({
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test
                })
                
                st.success("Data split complete!")
                
                # Feature Correlation Heatmap
                st.subheader("Feature Correlation")
                correlation_matrix = X.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                st.pyplot(fig)
                
                # Train/Test Split Visualization
                st.subheader("Train/Test Split")
                pie = pd.DataFrame({'Set': ['Train', 'Test'], 'Size': [len(X_train), len(X_test)]})
                fig = px.pie(pie, names='Set', values='Size', title="Train/Test Split")
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error during feature selection: {str(e)}")
    else:
        st.warning("Load and preprocess data first.")

# -------------- 🧠 Train Model --------------
elif page == "🧠 Train Model":
    st.title("🧠 Train Model")
    
    if st.session_state.X_train is None:
        st.warning("Please select features and split data first.")
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Support Vector Regressor (SVR)": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor()
        }
        
        model_name = st.selectbox("Choose Model", list(models.keys()))
        
        # Model parameters
        st.subheader("Model Parameters")
        if model_name == "Random Forest":
            n_estimators = st.slider("Number of Trees", 10, 200, 100)
            max_depth = st.slider("Max Depth", 1, 20, 10)
            models[model_name] = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        elif model_name == "SVR":
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            C = st.slider("C", 0.1, 10.0, 1.0)
            models[model_name] = SVR(kernel=kernel, C=C)
        elif model_name == "K-Nearest Neighbors":
            n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
            models[model_name] = KNeighborsRegressor(n_neighbors=n_neighbors)
        
        if st.button("Train Model"):
            try:
                model = models[model_name]
                model.fit(st.session_state.X_train, st.session_state.y_train)
                
                # Make predictions
                y_pred = model.predict(st.session_state.X_test)
                
                # Calculate metrics
                r2 = r2_score(st.session_state.y_test, y_pred)
                mse = mean_squared_error(st.session_state.y_test, y_pred)
                mae = mean_absolute_error(st.session_state.y_test, y_pred)
                
                # Store results
                st.session_state.model = model
                st.session_state.model_name = model_name
                st.session_state.y_pred = y_pred
                st.session_state.model_results[model_name] = {
                    "r2": r2,
                    "mse": mse,
                    "mae": mae
                }
                
                # Save model
                save_model(model, model_name)
                
                st.success(f"{model_name} trained successfully!")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R2 Score", f"{r2:.4f}")
                with col2:
                    st.metric("MSE", f"{mse:.4f}")
                with col3:
                    st.metric("MAE", f"{mae:.4f}")
                
                # Actual vs Predicted plot
                st.subheader("Actual vs Predicted")
                result_df = pd.DataFrame({
                    "Actual": st.session_state.y_test,
                    "Predicted": y_pred
                })
                fig = px.scatter(result_df, x="Actual", y="Predicted", 
                               title=f"{model_name} - Actual vs Predicted")
                fig.add_trace(go.Scatter(x=result_df["Actual"], y=result_df["Actual"],
                                       mode='lines', name='Perfect Prediction'))
                st.plotly_chart(fig)
                
                # Feature importance for tree-based models
                if model_name in ["Random Forest", "Decision Tree"]:
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame({
                        "Feature": st.session_state.X_train.columns,
                        "Importance": model.feature_importances_
                    }).sort_values("Importance", ascending=False)
                    fig = px.bar(importance_df, x="Feature", y="Importance",
                               title="Feature Importance")
                    st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")

# -------------- 📈 Results --------------
elif page == "📈 Results":
    st.title("📈 Model Results")
    
    if not st.session_state.model_results:
        st.warning("No models have been trained yet.")
    else:
        # Model comparison
        st.subheader("Model Comparison")
        results_df = pd.DataFrame(st.session_state.model_results).T
        st.dataframe(results_df)
        
        # Visualize model comparison
        fig = go.Figure()
        for metric in ["r2", "mse", "mae"]:
            fig.add_trace(go.Bar(
                x=results_df.index,
                y=results_df[metric],
                name=metric.upper()
            ))
        fig.update_layout(barmode='group', title="Model Comparison")
        st.plotly_chart(fig)
        
        # Show detailed results for selected model
        selected_model = st.selectbox("Select Model for Detailed Results", 
                                    list(st.session_state.model_results.keys()))
        
        if selected_model:
            results = st.session_state.model_results[selected_model]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R2 Score", f"{results['r2']:.4f}")
            with col2:
                st.metric("MSE", f"{results['mse']:.4f}")
            with col3:
                st.metric("MAE", f"{results['mae']:.4f}")
            
            # Load the model for predictions
            model = load_model(selected_model)
            if model:
                st.subheader("Make New Predictions")
                input_data = {}
                for feature in st.session_state.X_train.columns:
                    input_data[feature] = st.number_input(f"Enter {feature}", 
                                                        value=float(st.session_state.X_train[feature].mean()))
                
                if st.button("Predict"):
                    try:
                        input_df = pd.DataFrame([input_data])
                        prediction = model.predict(input_df)[0]
                        st.success(f"Predicted Value: {prediction:.4f}")
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
