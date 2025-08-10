import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🧠 Migraine Disease Progression Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None

# Main header
st.markdown('<h1 class="main-header">🧠 Migraine Disease Progression Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.selectbox("Choose a page:", [
    "🏠 Home & Data Upload",
    "📊 Data Analysis",
    "🤖 Model Training",
    "🔮 Predictions",
    "👤 Personal Insights",
    "📅 Prediction Calendar"
])

# ============================================================================
# CORE FUNCTIONS (Simplified for MVP)
# ============================================================================

@st.cache_data
def generate_sample_data():
    """Generate sample migraine data for demonstration"""
    np.random.seed(42)
    
    # Parameters
    n_users = 20
    days_per_user = 90
    
    data = []
    
    for user_id in range(1, n_users + 1):
        # User-specific patterns
        base_migraine_rate = np.random.uniform(0.05, 0.25)
        stress_sensitivity = np.random.uniform(0.5, 2.0)
        sleep_sensitivity = np.random.uniform(0.5, 2.0)
        
        start_date = datetime(2024, 1, 1)
        
        for day in range(days_per_user):
            current_date = start_date + timedelta(days=day)
            
            # Generate features with realistic patterns
            day_of_week = current_date.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Sleep patterns (worse on weekends for some)
            if is_weekend:
                sleep_hours = np.random.normal(7.5, 1.5)
            else:
                sleep_hours = np.random.normal(7.0, 1.0)
            sleep_hours = np.clip(sleep_hours, 4, 12)
            
            # Stress patterns (higher on weekdays)
            if is_weekend:
                stress_level = np.random.normal(3, 1.5)
            else:
                stress_level = np.random.normal(5, 2)
            stress_level = np.clip(stress_level, 1, 10)
            
            # Other features
            screen_time_hours = np.random.normal(6, 2)
            screen_time_hours = np.clip(screen_time_hours, 1, 12)
            
            hydration_glasses = np.random.normal(6, 2)
            hydration_glasses = np.clip(hydration_glasses, 1, 15)
            
            exercise_minutes = np.random.exponential(20)
            exercise_minutes = np.clip(exercise_minutes, 0, 120)
            
            # Migraine probability based on risk factors
            migraine_prob = base_migraine_rate
            
            # Risk factors
            if sleep_hours < 6:
                migraine_prob += 0.15 * sleep_sensitivity
            if stress_level > 7:
                migraine_prob += 0.12 * stress_sensitivity
            if hydration_glasses < 4:
                migraine_prob += 0.08
            if screen_time_hours > 8:
                migraine_prob += 0.06
            if exercise_minutes < 10:
                migraine_prob += 0.05
            
            # Weekend effect for some users
            if is_weekend and user_id % 3 == 0:
                migraine_prob += 0.1
            
            # Seasonal effect
            if current_date.month in [3, 4, 9, 10]:  # Spring/Fall
                migraine_prob += 0.05
            
            migraine_prob = np.clip(migraine_prob, 0, 0.8)
            has_migraine = np.random.binomial(1, migraine_prob)
            
            # Migraine severity if migraine occurs
            if has_migraine:
                migraine_severity = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
            else:
                migraine_severity = 0
            
            data.append({
                'user_id': user_id,
                'date': current_date.strftime('%Y-%m-%d'),
                'sleep_hours': round(sleep_hours, 1),
                'stress_level': round(stress_level, 1),
                'screen_time_hours': round(screen_time_hours, 1),
                'hydration_glasses': round(hydration_glasses, 1),
                'exercise_minutes': round(exercise_minutes, 0),
                'has_migraine': has_migraine,
                'migraine_severity': migraine_severity
            })
    
    return pd.DataFrame(data)

def preprocess_dates(df):
    """Convert date column to datetime and create temporal features"""
    df_processed = df.copy()
    
    # Convert date column to datetime
    df_processed['date'] = pd.to_datetime(df_processed['date'])
    
    # Create day number for easier processing
    min_date = df_processed['date'].min()
    df_processed['day'] = (df_processed['date'] - min_date).dt.days
    
    # Create additional date-based features
    df_processed['year'] = df_processed['date'].dt.year
    df_processed['month'] = df_processed['date'].dt.month
    df_processed['day_of_week'] = df_processed['date'].dt.dayofweek
    df_processed['day_of_year'] = df_processed['date'].dt.dayofyear
    df_processed['week_of_year'] = df_processed['date'].dt.isocalendar().week
    df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
    
    # Seasonal patterns
    df_processed['season'] = df_processed['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Cyclical encoding for temporal features
    df_processed['day_of_week_sin'] = np.sin(2 * np.pi * df_processed['day_of_week'] / 7)
    df_processed['day_of_week_cos'] = np.cos(2 * np.pi * df_processed['day_of_week'] / 7)
    df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
    df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)
    
    return df_processed

def create_disease_progression_features(df):
    """Create features that capture migraine disease progression patterns"""
    df_prog = preprocess_dates(df)
    df_prog = df_prog.sort_values(['user_id', 'date']).reset_index(drop=True)
    
    # Initialize progression features
    df_prog['days_since_last_migraine'] = 0
    df_prog['migraine_frequency_7d'] = 0
    df_prog['migraine_frequency_30d'] = 0
    df_prog['stress_burden_7d'] = 0
    df_prog['sleep_debt'] = np.maximum(0, 7.5 - df_prog['sleep_hours'])
    df_prog['cumulative_sleep_debt_7d'] = 0
    
    # Calculate features for each user
    for user_id in df_prog['user_id'].unique():
        user_mask = df_prog['user_id'] == user_id
        user_data = df_prog[user_mask].copy().sort_values('date')
        
        # Days since last migraine
        migraine_dates = user_data[user_data['has_migraine'] == 1]['date'].values
        days_since = []
        
        for current_date in user_data['date'].values:
            past_migraines = migraine_dates[migraine_dates < current_date]
            if len(past_migraines) > 0:
                last_migraine = past_migraines[-1]
                days_since.append((pd.to_datetime(current_date) - pd.to_datetime(last_migraine)).days)
            else:
                days_since.append(999)
        
        df_prog.loc[user_mask, 'days_since_last_migraine'] = days_since
        
        # Rolling features
        user_indices = user_data.index
        
        # 7-day rolling migraine frequency
        migraine_7d = df_prog.loc[user_indices, 'has_migraine'].rolling(window=7, min_periods=1).sum()
        df_prog.loc[user_indices, 'migraine_frequency_7d'] = migraine_7d
        
        # 30-day rolling migraine frequency  
        migraine_30d = df_prog.loc[user_indices, 'has_migraine'].rolling(window=30, min_periods=1).sum()
        df_prog.loc[user_indices, 'migraine_frequency_30d'] = migraine_30d
        
        # 7-day stress burden
        stress_7d = df_prog.loc[user_indices, 'stress_level'].rolling(window=7, min_periods=1).sum()
        df_prog.loc[user_indices, 'stress_burden_7d'] = stress_7d
        
        # 7-day cumulative sleep debt
        sleep_debt_7d = df_prog.loc[user_indices, 'sleep_debt'].rolling(window=7, min_periods=1).sum()
        df_prog.loc[user_indices, 'cumulative_sleep_debt_7d'] = sleep_debt_7d
    
    # Additional risk factors
    df_prog['dehydration_risk'] = (df_prog['hydration_glasses'] < 6).astype(int)
    df_prog['high_stress_risk'] = (df_prog['stress_level'] > 7).astype(int)
    df_prog['poor_sleep_risk'] = (df_prog['sleep_hours'] < 6.5).astype(int)
    df_prog['high_screen_risk'] = (df_prog['screen_time_hours'] > 8).astype(int)
    df_prog['low_exercise_risk'] = (df_prog['exercise_minutes'] < 30).astype(int)
    
    # Combined risk score
    df_prog['risk_score'] = (
        df_prog['dehydration_risk'] +
        df_prog['high_stress_risk'] +
        df_prog['poor_sleep_risk'] +
        df_prog['high_screen_risk'] +
        df_prog['low_exercise_risk']
    )
    
    return df_prog

def prepare_model_data(df_prog):
    """Prepare data for model training"""
    # Select features for modeling
    feature_cols = [
        'sleep_hours', 'stress_level', 'screen_time_hours', 'hydration_glasses', 'exercise_minutes',
        'day_of_week', 'month', 'is_weekend', 'days_since_last_migraine',
        'migraine_frequency_7d', 'migraine_frequency_30d', 'stress_burden_7d', 'cumulative_sleep_debt_7d',
        'dehydration_risk', 'high_stress_risk', 'poor_sleep_risk', 'high_screen_risk', 'low_exercise_risk',
        'risk_score', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos'
    ]
    
    # Ensure all feature columns exist
    available_features = [col for col in feature_cols if col in df_prog.columns]
    
    X = df_prog[available_features]
    y = df_prog['has_migraine']
    
    return X, y, available_features

def train_models(X, y):
    """Train multiple models and return results"""
    # Split data temporally
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(50, 25), random_state=42, max_iter=500)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        if name == 'Neural Network':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'f1_score': f1,
            'roc_auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'scaler': scaler if name == 'Neural Network' else None
        }
    
    return results, X_test, y_test

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def home_page():
    st.markdown("## 🏠 Welcome to the Migraine Disease Progression Predictor")
    
    st.markdown("""
    This application uses advanced machine learning to predict migraine episodes based on temporal patterns 
    and disease progression indicators. It treats migraine prediction as a **disease progression problem** 
    rather than isolated daily predictions.
    
    ### 🎯 Key Features:
    - **Temporal Analysis**: Uses date-based features and progression patterns
    - **Early Warning**: Provides advance predictions based on risk accumulation
    - **Personal Insights**: Generates personalized recommendations for each user
    - **Multiple Models**: Compares different ML approaches for optimal results
    """)
    
    st.markdown("---")
    
    # Data upload section
    st.markdown("### 📁 Data Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your migraine data (CSV format)",
            type=['csv'],
            help="CSV should contain: user_id, date, sleep_hours, stress_level, screen_time_hours, hydration_glasses, exercise_minutes, has_migraine"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_cols = ['user_id', 'date', 'sleep_hours', 'stress_level', 'screen_time_hours', 
                               'hydration_glasses', 'exercise_minutes', 'has_migraine']
                
                if all(col in df.columns for col in required_cols):
                    st.session_state.df_processed = df
                    st.session_state.data_loaded = True
                    st.success(f"✅ Data loaded successfully! {len(df)} records from {df['user_id'].nunique()} users")
                else:
                    st.error(f"❌ Missing required columns. Expected: {required_cols}")
                    
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    with col2:
        st.markdown("### 🎲 Demo Data")
        if st.button("Generate Sample Data", type="primary"):
            with st.spinner("Generating sample data..."):
                df_sample = generate_sample_data()
                st.session_state.df_processed = df_sample
                st.session_state.data_loaded = True
                st.success("✅ Sample data generated!")
    
    if st.session_state.data_loaded:
        st.markdown("---")
        st.markdown("### 📊 Data Preview")
        
        df = st.session_state.df_processed
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Unique Users", df['user_id'].nunique())
        with col3:
            st.metric("Date Range", f"{df['date'].nunique()} days")
        with col4:
            st.metric("Migraine Rate", f"{df['has_migraine'].mean():.1%}")
        
        st.dataframe(df.head(), use_container_width=True)

def data_analysis_page():
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first on the Home page.")
        return
    
    st.markdown("## 📊 Data Analysis & Patterns")
    
    df = st.session_state.df_processed
    df_prog = create_disease_progression_features(df)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Days", len(df_prog))
    with col2:
        st.metric("Users", df_prog['user_id'].nunique())
    with col3:
        st.metric("Migraine Days", df_prog['has_migraine'].sum())
    with col4:
        st.metric("Overall Rate", f"{df_prog['has_migraine'].mean():.1%}")
    
    st.markdown("---")
    
    # Temporal patterns
    st.markdown("### ⏰ Temporal Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Day of week pattern
        dow_data = df_prog.groupby('day_of_week')['has_migraine'].mean().reset_index()
        dow_data['day_name'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig = px.bar(dow_data, x='day_name', y='has_migraine', 
                     title='Migraine Frequency by Day of Week',
                     labels={'has_migraine': 'Migraine Rate', 'day_name': 'Day'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly pattern
        monthly_data = df_prog.groupby('month')['has_migraine'].mean().reset_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_data['month_name'] = [month_names[i-1] for i in monthly_data['month']]
        
        fig = px.line(monthly_data, x='month_name', y='has_migraine',
                     title='Migraine Frequency by Month',
                     labels={'has_migraine': 'Migraine Rate', 'month_name': 'Month'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Risk factor analysis
    st.markdown("### ⚡ Risk Factor Analysis")
    
    # Correlation heatmap
    numeric_cols = ['sleep_hours', 'stress_level', 'screen_time_hours', 'hydration_glasses', 
                   'exercise_minutes', 'has_migraine']
    corr_matrix = df_prog[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                    title='Feature Correlation Matrix',
                    color_continuous_scale='RdBu_r',
                    aspect='auto')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Sleep vs migraine
        sleep_bins = pd.cut(df_prog['sleep_hours'], bins=4, labels=['<6h', '6-7h', '7-8h', '>8h'])
        sleep_migraine = df_prog.groupby(sleep_bins)['has_migraine'].mean().reset_index()
        
        fig = px.bar(sleep_migraine, x='sleep_hours', y='has_migraine',
                     title='Migraine Rate by Sleep Duration',
                     labels={'has_migraine': 'Migraine Rate', 'sleep_hours': 'Sleep Hours'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Stress vs migraine
        stress_bins = pd.cut(df_prog['stress_level'], bins=4, labels=['Low', 'Medium', 'High', 'Very High'])
        stress_migraine = df_prog.groupby(stress_bins)['has_migraine'].mean().reset_index()
        
        fig = px.bar(stress_migraine, x='stress_level', y='has_migraine',
                     title='Migraine Rate by Stress Level',
                     labels={'has_migraine': 'Migraine Rate', 'stress_level': 'Stress Level'})
        st.plotly_chart(fig, use_container_width=True)

def model_training_page():
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first on the Home page.")
        return
    
    st.markdown("## 🤖 Model Training & Evaluation")
    
    df = st.session_state.df_processed
    
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            # Create progression features
            df_prog = create_disease_progression_features(df)
            
            # Prepare model data
            X, y, feature_names = prepare_model_data(df_prog)
            
            # Train models
            results, X_test, y_test = train_models(X, y)
            
            # Store results in session state
            st.session_state.model_results = results
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.feature_names = feature_names
            st.session_state.df_prog = df_prog
            st.session_state.model_trained = True
            
            st.success("✅ Models trained successfully!")
    
    if st.session_state.model_trained:
        st.markdown("---")
        
        # Model comparison
        st.markdown("### 📈 Model Performance Comparison")
        
        results = st.session_state.model_results
        
        # Create comparison DataFrame
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'F1-Score': result['f1_score'],
                'ROC-AUC': result['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(comparison_df, x='Model', y='F1-Score',
                        title='F1-Score Comparison',
                        color='F1-Score',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(comparison_df, x='Model', y='ROC-AUC',
                        title='ROC-AUC Comparison',
                        color='ROC-AUC',
                        color_continuous_scale='plasma')
            st.plotly_chart(fig, use_container_width=True)
        
        # Best model details
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        best_result = results[best_model_name]
        
        st.markdown(f"### 🏆 Best Model: {best_model_name}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("F1-Score", f"{best_result['f1_score']:.3f}")
        with col2:
            st.metric("ROC-AUC", f"{best_result['roc_auc']:.3f}")
        with col3:
            accuracy = (best_result['predictions'] == st.session_state.y_test).mean()
            st.metric("Accuracy", f"{accuracy:.3f}")
        
        # Confusion Matrix
        cm = confusion_matrix(st.session_state.y_test, best_result['predictions'])
        
        fig = px.imshow(cm, 
                       title='Confusion Matrix',
                       labels=dict(x="Predicted", y="Actual"),
                       x=['No Migraine', 'Migraine'],
                       y=['No Migraine', 'Migraine'],
                       color_continuous_scale='Blues',
                       text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (for tree-based models)
        if hasattr(best_result['model'], 'feature_importances_'):
            st.markdown("### 🔍 Feature Importance")
            
            importance_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': best_result['model'].feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(importance_df, x='Importance', y='Feature',
                        title='Top 10 Most Important Features',
                        orientation='h')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

def predictions_page():
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first on the Model Training page.")
        return
    
    st.markdown("## 🔮 Migraine Predictions")
    
    df_prog = st.session_state.df_prog
    results = st.session_state.model_results
    
    # Select best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_result = results[best_model_name]
    
    st.markdown(f"### Using: {best_model_name}")
    
    # Single prediction section
    st.markdown("### 🎯 Single Day Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sleep_input = st.slider("Sleep Hours", 4.0, 12.0, 7.5, 0.1)
        stress_input = st.slider("Stress Level (1-10)", 1.0, 10.0, 5.0, 0.1)
        screen_input = st.slider("Screen Time Hours", 1.0, 12.0, 6.0, 0.1)
        hydration_input = st.slider("Hydration (glasses)", 1.0, 15.0, 8.0, 0.1)
    
    with col2:
        exercise_input = st.slider("Exercise Minutes", 0.0, 120.0, 30.0, 1.0)
        day_of_week_input = st.selectbox("Day of Week", 
                                         ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                          'Friday', 'Saturday', 'Sunday'])
        
        # Convert day name to number
        day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                      'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        dow_num = day_mapping[day_of_week_input]
        
        month_input = st.selectbox("Month", 
                                  ['January', 'February', 'March', 'April', 'May', 'June',
                                   'July', 'August', 'September', 'October', 'November', 'December'])
        
        month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4,
                        'May': 5, 'June': 6, 'July': 7, 'August': 8,
                        'September': 9, 'October': 10, 'November': 11, 'December': 12}
        month_num = month_mapping[month_input]
    
    if st.button("🔮 Predict Migraine Risk", type="primary"):
        # Create prediction input
        prediction_input = {
            'sleep_hours': sleep_input,
            'stress_level': stress_input,
            'screen_time_hours': screen_input,
            'hydration_glasses': hydration_input,
            'exercise_minutes': exercise_input,
            'day_of_week': dow_num,
            'month': month_num,
            'is_weekend': 1 if dow_num >= 5 else 0,
            'days_since_last_migraine': 7,  # Default assumption
            'migraine_frequency_7d': 1,     # Default assumption
            'migraine_frequency_30d': 3,    # Default assumption
            'stress_burden_7d': stress_input * 7,
            'cumulative_sleep_debt_7d': max(0, 7.5 - sleep_input) * 7,
            'dehydration_risk': 1 if hydration_input < 6 else 0,
            'high_stress_risk': 1 if stress_input > 7 else 0,
            'poor_sleep_risk': 1 if sleep_input < 6.5 else 0,
            'high_screen_risk': 1 if screen_input > 8 else 0,
            'low_exercise_risk': 1 if exercise_input < 30 else 0,
            'day_of_week_sin': np.sin(2 * np.pi * dow_num / 7),
            'day_of_week_cos': np.cos(2 * np.pi * dow_num / 7),
            'month_sin': np.sin(2 * np.pi * month_num / 12),
            'month_cos': np.cos(2 * np.pi * month_num / 12)
        }
        
        # Calculate risk score
        prediction_input['risk_score'] = (
            prediction_input['dehydration_risk'] +
            prediction_input['high_stress_risk'] +
            prediction_input['poor_sleep_risk'] +
            prediction_input['high_screen_risk'] +
            prediction_input['low_exercise_risk']
        )
        
        # Create DataFrame for prediction
        pred_df = pd.DataFrame([prediction_input])
        
        # Ensure all required features are present
        for feature in st.session_state.feature_names:
            if feature not in pred_df.columns:
                pred_df[feature] = 0
        
        # Reorder columns to match training data
        pred_df = pred_df[st.session_state.feature_names]
        
        # Make prediction
        if best_result['scaler'] is not None:
            pred_scaled = best_result['scaler'].transform(pred_df)
            prediction_prob = best_result['model'].predict_proba(pred_scaled)[0, 1]
        else:
            prediction_prob = best_result['model'].predict_proba(pred_df)[0, 1]
        
        prediction = "High Risk" if prediction_prob > 0.5 else "Low Risk"
        
        # Display prediction
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction_prob > 0.7:
                st.error(f"🚨 **HIGH RISK**: {prediction_prob:.1%} chance of migraine")
            elif prediction_prob > 0.4:
                st.warning(f"⚠️ **MODERATE RISK**: {prediction_prob:.1%} chance of migraine")
            else:
                st.success(f"✅ **LOW RISK**: {prediction_prob:.1%} chance of migraine")
        
        with col2:
            # Risk factors breakdown
            st.markdown("**Risk Factors:**")
            risk_factors = []
            if prediction_input['poor_sleep_risk']:
                risk_factors.append("😴 Poor Sleep")
            if prediction_input['high_stress_risk']:
                risk_factors.append("😰 High Stress")
            if prediction_input['dehydration_risk']:
                risk_factors.append("💧 Dehydration")
            if prediction_input['high_screen_risk']:
                risk_factors.append("📱 Excessive Screen Time")
            if prediction_input['low_exercise_risk']:
                risk_factors.append("🏃‍♂️ Low Exercise")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("✅ No major risk factors detected")

def personal_insights_page():
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first on the Model Training page.")
        return
    
    st.markdown("## 👤 Personal Insights & Recommendations")
    
    df_prog = st.session_state.df_prog
    
    # User selection
    users = sorted(df_prog['user_id'].unique())
    selected_user = st.selectbox("Select User for Analysis:", users)
    
    # Filter data for selected user
    user_data = df_prog[df_prog['user_id'] == selected_user].copy()
    user_data = user_data.sort_values('date').reset_index(drop=True)
    
    # Convert date strings to datetime if needed
    if user_data['date'].dtype == 'object':
        user_data['date'] = pd.to_datetime(user_data['date'])
    
    st.markdown(f"### Analysis for User {selected_user}")
    
    # User statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_days = len(user_data)
        st.metric("Total Days", total_days)
    
    with col2:
        migraine_days = user_data['has_migraine'].sum()
        st.metric("Migraine Days", migraine_days)
    
    with col3:
        migraine_rate = user_data['has_migraine'].mean()
        st.metric("Migraine Rate", f"{migraine_rate:.1%}")
    
    with col4:
        avg_severity = user_data[user_data['has_migraine'] == 1]['migraine_severity'].mean()
        st.metric("Avg Severity", f"{avg_severity:.1f}" if not pd.isna(avg_severity) else "N/A")
    
    st.markdown("---")
    
    # Timeline visualization
    st.markdown("### 📅 Migraine Timeline")
    
    fig = go.Figure()
    
    # Add migraine days
    migraine_days = user_data[user_data['has_migraine'] == 1]
    fig.add_trace(go.Scatter(
        x=migraine_days['date'],
        y=migraine_days['migraine_severity'],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Migraine Days',
        hovertemplate='Date: %{x}<br>Severity: %{y}<extra></extra>'
    ))
    
    # Add trend line for risk score
    fig.add_trace(go.Scatter(
        x=user_data['date'],
        y=user_data['risk_score'],
        mode='lines',
        line=dict(color='orange', width=2),
        name='Risk Score',
        yaxis='y2',
        hovertemplate='Date: %{x}<br>Risk Score: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Migraine Timeline for User {selected_user}',
        xaxis_title='Date',
        yaxis_title='Migraine Severity',
        yaxis2=dict(title='Risk Score', overlaying='y', side='right'),
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Pattern analysis
    st.markdown("### 🔍 Personal Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sleep pattern
        sleep_migraine = user_data.groupby(pd.cut(user_data['sleep_hours'], 
                                                 bins=4, labels=['<6h', '6-7h', '7-8h', '>8h']))['has_migraine'].mean()
        
        fig = px.bar(x=sleep_migraine.index.astype(str), y=sleep_migraine.values,
                     title='Your Migraine Rate by Sleep Duration',
                     labels={'x': 'Sleep Hours', 'y': 'Migraine Rate'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Stress pattern
        stress_migraine = user_data.groupby(pd.cut(user_data['stress_level'], 
                                                  bins=4, labels=['Low', 'Medium', 'High', 'Very High']))['has_migraine'].mean()
        
        fig = px.bar(x=stress_migraine.index.astype(str), y=stress_migraine.values,
                     title='Your Migraine Rate by Stress Level',
                     labels={'x': 'Stress Level', 'y': 'Migraine Rate'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("---")
    st.markdown("### 💡 Personalized Recommendations")
    
    recommendations = []
    
    # Sleep recommendations
    avg_sleep = user_data['sleep_hours'].mean()
    migraine_sleep = user_data[user_data['has_migraine'] == 1]['sleep_hours'].mean()
    
    if avg_sleep < 7:
        recommendations.append({
            'category': '😴 Sleep',
            'recommendation': f'Increase sleep duration. Your average is {avg_sleep:.1f}h, aim for 7-8 hours.',
            'priority': 'High'
        })
    elif not pd.isna(migraine_sleep) and migraine_sleep < avg_sleep:
        recommendations.append({
            'category': '😴 Sleep',
            'recommendation': f'Maintain consistent sleep. Migraines occur more often with {migraine_sleep:.1f}h vs your average {avg_sleep:.1f}h.',
            'priority': 'Medium'
        })
    
    # Stress recommendations
    avg_stress = user_data['stress_level'].mean()
    migraine_stress = user_data[user_data['has_migraine'] == 1]['stress_level'].mean()
    
    if avg_stress > 6:
        recommendations.append({
            'category': '😰 Stress',
            'recommendation': f'Focus on stress management. Your average stress is {avg_stress:.1f}/10.',
            'priority': 'High'
        })
    elif not pd.isna(migraine_stress) and migraine_stress > avg_stress:
        recommendations.append({
            'category': '😰 Stress',
            'recommendation': f'Monitor stress levels. Migraines occur with higher stress ({migraine_stress:.1f} vs {avg_stress:.1f}).',
            'priority': 'Medium'
        })
    
    # Hydration recommendations
    avg_hydration = user_data['hydration_glasses'].mean()
    if avg_hydration < 8:
        recommendations.append({
            'category': '💧 Hydration',
            'recommendation': f'Increase water intake. Current average: {avg_hydration:.1f} glasses, aim for 8-10.',
            'priority': 'Medium'
        })
    
    # Exercise recommendations
    avg_exercise = user_data['exercise_minutes'].mean()
    if avg_exercise < 30:
        recommendations.append({
            'category': '🏃‍♂️ Exercise',
            'recommendation': f'Increase physical activity. Current average: {avg_exercise:.0f} minutes, aim for 30+ minutes.',
            'priority': 'Medium'
        })
    
    # Screen time recommendations
    avg_screen = user_data['screen_time_hours'].mean()
    if avg_screen > 8:
        recommendations.append({
            'category': '📱 Screen Time',
            'recommendation': f'Reduce screen exposure. Current average: {avg_screen:.1f} hours, consider breaks and blue light filters.',
            'priority': 'Low'
        })
    
    # Display recommendations
    if recommendations:
        for rec in recommendations:
            priority_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
            st.markdown(f"""
            <div class="metric-card">
                <strong>{priority_color[rec['priority']]} {rec['category']}</strong><br>
                {rec['recommendation']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("🎉 Great job! Your current patterns look healthy. Keep up the good work!")

def prediction_calendar_page():
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first on the Model Training page.")
        return
    
    st.markdown("## 📅 Prediction Calendar")
    st.markdown("Generate migraine risk predictions for upcoming days based on planned activities.")
    
    df_prog = st.session_state.df_prog
    results = st.session_state.model_results
    
    # Select best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_result = results[best_model_name]
    
    # Date range selection
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", datetime.now().date())
    
    with col2:
        num_days = st.slider("Number of Days", 1, 30, 7)
    
    st.markdown("### 📋 Daily Plan Input")
    st.markdown("Enter your planned values for each lifestyle factor:")
    
    # Default values
    default_sleep = st.slider("Default Sleep Hours", 4.0, 12.0, 7.5, 0.1)
    default_stress = st.slider("Default Stress Level", 1.0, 10.0, 5.0, 0.1)
    default_screen = st.slider("Default Screen Time", 1.0, 12.0, 6.0, 0.1)
    default_hydration = st.slider("Default Hydration", 1.0, 15.0, 8.0, 0.1)
    default_exercise = st.slider("Default Exercise Minutes", 0.0, 120.0, 30.0, 1.0)
    
    if st.button("🔮 Generate Calendar Predictions", type="primary"):
        predictions = []
        
        for i in range(num_days):
            current_date = start_date + timedelta(days=i)
            dow_num = current_date.weekday()
            month_num = current_date.month
            
            # Create prediction input
            prediction_input = {
                'sleep_hours': default_sleep,
                'stress_level': default_stress,
                'screen_time_hours': default_screen,
                'hydration_glasses': default_hydration,
                'exercise_minutes': default_exercise,
                'day_of_week': dow_num,
                'month': month_num,
                'is_weekend': 1 if dow_num >= 5 else 0,
                'days_since_last_migraine': max(1, i + 1),  # Assume no recent migraines
                'migraine_frequency_7d': 0,  # Assume no recent migraines
                'migraine_frequency_30d': 1,  # Assume low frequency
                'stress_burden_7d': default_stress * 7,
                'cumulative_sleep_debt_7d': max(0, 7.5 - default_sleep) * 7,
                'dehydration_risk': 1 if default_hydration < 6 else 0,
                'high_stress_risk': 1 if default_stress > 7 else 0,
                'poor_sleep_risk': 1 if default_sleep < 6.5 else 0,
                'high_screen_risk': 1 if default_screen > 8 else 0,
                'low_exercise_risk': 1 if default_exercise < 30 else 0,
                'day_of_week_sin': np.sin(2 * np.pi * dow_num / 7),
                'day_of_week_cos': np.cos(2 * np.pi * dow_num / 7),
                'month_sin': np.sin(2 * np.pi * month_num / 12),
                'month_cos': np.cos(2 * np.pi * month_num / 12)
            }
            
            # Calculate risk score
            prediction_input['risk_score'] = (
                prediction_input['dehydration_risk'] +
                prediction_input['high_stress_risk'] +
                prediction_input['poor_sleep_risk'] +
                prediction_input['high_screen_risk'] +
                prediction_input['low_exercise_risk']
            )
            
            # Create DataFrame for prediction
            pred_df = pd.DataFrame([prediction_input])
            
            # Ensure all required features are present
            for feature in st.session_state.feature_names:
                if feature not in pred_df.columns:
                    pred_df[feature] = 0
            
            # Reorder columns to match training data
            pred_df = pred_df[st.session_state.feature_names]
            
            # Make prediction
            if best_result['scaler'] is not None:
                pred_scaled = best_result['scaler'].transform(pred_df)
                prediction_prob = best_result['model'].predict_proba(pred_scaled)[0, 1]
            else:
                prediction_prob = best_result['model'].predict_proba(pred_df)[0, 1]
            
            predictions.append({
                'Date': current_date.strftime('%Y-%m-%d'),
                'Day': current_date.strftime('%A'),
                'Risk_Probability': prediction_prob,
                'Risk_Level': 'High' if prediction_prob > 0.6 else 'Medium' if prediction_prob > 0.3 else 'Low',
                'Risk_Score': prediction_input['risk_score']
            })
        
        # Create predictions DataFrame
        pred_df = pd.DataFrame(predictions)
        
        # Display calendar
        st.markdown("### 📊 Risk Calendar")
        
        # Calendar visualization
        fig = px.bar(pred_df, x='Date', y='Risk_Probability', 
                     color='Risk_Level',
                     color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
                     title='Daily Migraine Risk Predictions',
                     labels={'Risk_Probability': 'Migraine Risk Probability'})
        
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk summary table
        st.markdown("### 📋 Detailed Predictions")
        
        # Color-code the risk levels
        def color_risk_level(val):
            if val == 'High':
                return 'background-color: #ffebee'
            elif val == 'Medium':
                return 'background-color: #fff3e0'
            else:
                return 'background-color: #e8f5e8'
        
        # Format probability as percentage
        pred_df['Risk_Probability_Pct'] = (pred_df['Risk_Probability'] * 100).round(1).astype(str) + '%'
        
        display_df = pred_df[['Date', 'Day', 'Risk_Probability_Pct', 'Risk_Level']].copy()
        display_df.columns = ['Date', 'Day of Week', 'Risk Probability', 'Risk Level']
        
        styled_df = display_df.style.applymap(color_risk_level, subset=['Risk Level'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Summary statistics
        high_risk_days = sum(pred_df['Risk_Level'] == 'High')
        medium_risk_days = sum(pred_df['Risk_Level'] == 'Medium')
        low_risk_days = sum(pred_df['Risk_Level'] == 'Low')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔴 High Risk Days", high_risk_days)
        with col2:
            st.metric("🟡 Medium Risk Days", medium_risk_days)
        with col3:
            st.metric("🟢 Low Risk Days", low_risk_days)
        
        # Recommendations based on calendar
        if high_risk_days > 0:
            st.markdown("### ⚠️ High Risk Day Recommendations")
            st.markdown("""
            For high-risk days, consider:
            - 😴 Prioritize 8+ hours of sleep the night before
            - 🧘‍♀️ Practice stress reduction techniques
            - 💧 Increase water intake throughout the day
            - 📱 Limit screen time and take regular breaks
            - 🏃‍♂️ Engage in light exercise or walking
            - 💊 Have migraine medication readily available
            """)

# ============================================================================
# MAIN APP ROUTING
# ============================================================================

# Route to appropriate page
if page == "🏠 Home & Data Upload":
    home_page()
elif page == "📊 Data Analysis":
    data_analysis_page()
elif page == "🤖 Model Training":
    model_training_page()
elif page == "🔮 Predictions":
    predictions_page()
elif page == "👤 Personal Insights":
    personal_insights_page()
elif page == "📅 Prediction Calendar":
    prediction_calendar_page()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🧠 Migraine Disease Progression Predictor</p>
    <p><em>Built with Streamlit • Machine Learning for Healthcare</em></p>
    <p style="font-size: 0.8em;">⚠️ This tool is for informational purposes only and should not replace professional medical advice.</p>
</div>
""", unsafe_allow_html=True)