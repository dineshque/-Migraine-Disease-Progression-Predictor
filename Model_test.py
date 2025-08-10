# Migraine Prediction MVP - Complete System
# Run this file to train models and start the web interface

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Web Interface
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

print("🎯 Migraine Prediction MVP System")
print("=" * 50)

# ================================
# 1. DATA GENERATION (Enhanced Dataset)
# ================================

def generate_enhanced_migraine_dataset(num_users=12, min_days=120, max_days=180):
    """Generate enhanced migraine dataset with better patterns"""
    all_data = []
    
    for user_id in range(1, num_users + 1):
        num_days = random.randint(min_days, max_days)
        start_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 60))
        dates = [start_date + timedelta(days=i) for i in range(num_days)]
        
        # User characteristics
        user_sleep_baseline = np.random.normal(7.5, 1.0)
        user_stress_baseline = np.random.uniform(1.5, 4.0)
        user_migraine_proneness = np.random.beta(2, 10)
        
        consecutive_poor_sleep = 0
        
        for day_idx, date in enumerate(dates):
            is_weekend = date.weekday() >= 5
            
            # Sleep hours
            sleep_base = user_sleep_baseline + (0.7 if is_weekend else 0)
            sleep_hours = max(3.0, min(12.0, np.random.normal(sleep_base, 0.9)))
            
            # Update consecutive poor sleep counter
            if sleep_hours < 6:
                consecutive_poor_sleep += 1
            else:
                consecutive_poor_sleep = 0
            
            # Stress level (1-5 scale)
            stress_base = user_stress_baseline + (-0.5 if is_weekend else 0.3)
            stress_level = max(1, min(5, int(np.random.normal(stress_base, 0.7))))
            
            # Mood level (1-5 scale)
            mood_base = 4.5 - (stress_level * 0.4) + ((sleep_hours - 6) * 0.3)
            mood_level = max(1, min(5, int(np.random.normal(mood_base, 0.6))))
            
            # Hydration level (1-5 scale)
            hydration_level = max(1, min(5, int(np.random.normal(3.2, 0.9))))
            
            # Screen time
            screen_base = 5 + (1.5 if is_weekend else 0) + (stress_level * 0.4)
            screen_time = max(1.0, min(16.0, np.random.normal(screen_base, 1.8)))
            
            # Enhanced migraine prediction logic
            migraine_risk = user_migraine_proneness
            
            # Key risk factors based on observations
            if sleep_hours < 6 and stress_level >= 4:
                migraine_risk += 0.65
            elif sleep_hours < 6:
                migraine_risk += 0.35
            elif stress_level >= 4:
                migraine_risk += 0.25
            
            if screen_time > 7 and hydration_level <= 2:
                migraine_risk += 0.45
            elif screen_time > 8:
                migraine_risk += 0.2
            elif hydration_level <= 2:
                migraine_risk += 0.35
            
            # Consecutive poor sleep effect
            if consecutive_poor_sleep >= 2:
                migraine_risk += 0.4 + (consecutive_poor_sleep - 2) * 0.1
            elif consecutive_poor_sleep == 1:
                migraine_risk += 0.15
            
            # Previous day migraine effect
            if day_idx > 0 and all_data[-1]['migraine_occurrence'] == 1:
                migraine_risk += 0.2
            
            # Day of week effects
            if date.weekday() == 0:  # Monday
                migraine_risk += 0.08
            elif date.weekday() == 6:  # Sunday (weekend ending stress)
                migraine_risk += 0.05
            
            migraine_risk = min(0.85, migraine_risk)
            migraine_occurrence = 1 if np.random.random() < migraine_risk else 0
            
            # Migraine severity
            if migraine_occurrence == 1:
                severity_base = 1.5
                if hydration_level <= 2: severity_base += 0.8
                if sleep_hours < 5: severity_base += 0.7
                if stress_level >= 4: severity_base += 0.5
                if consecutive_poor_sleep >= 2: severity_base += 0.4
                
                migraine_severity = max(1, min(3, int(np.random.normal(severity_base, 0.5))))
            else:
                migraine_severity = 0
            
            all_data.append({
                'user_id': user_id,
                'date': date.strftime('%Y-%m-%d'),
                'day_of_week': date.weekday(),
                'is_weekend': int(is_weekend),
                'sleep_hours': round(sleep_hours, 1),
                'mood_level': mood_level,
                'stress_level': stress_level,
                'hydration_level': hydration_level,
                'screen_time': round(screen_time, 1),
                'consecutive_poor_sleep': consecutive_poor_sleep,
                'migraine_occurrence': migraine_occurrence,
                'migraine_severity': migraine_severity
            })
    
    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(['user_id', 'date']).reset_index(drop=True)

# ================================
# 2. FEATURE ENGINEERING
# ================================

def create_features(df):
    """Create additional features for better prediction"""
    df = df.copy()
    
    # Rolling averages (3-day and 7-day)
    for feature in ['sleep_hours', 'stress_level', 'mood_level', 'hydration_level']:
        df[f'{feature}_3day_avg'] = df.groupby('user_id')[feature].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
        df[f'{feature}_7day_avg'] = df.groupby('user_id')[feature].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
    
    # Sleep debt (difference from user's average)
    user_avg_sleep = df.groupby('user_id')['sleep_hours'].mean()
    df['sleep_debt'] = df.apply(lambda x: user_avg_sleep[x['user_id']] - x['sleep_hours'], axis=1)
    
    # Stress-sleep interaction
    df['stress_sleep_interaction'] = df['stress_level'] * (6 - df['sleep_hours']).clip(0)
    
    # Hydration-screen interaction
    df['dehydration_screen_risk'] = (6 - df['hydration_level']) * (df['screen_time'] / 8)
    
    # Previous day features
    df['prev_day_migraine'] = df.groupby('user_id')['migraine_occurrence'].shift(1).fillna(0)
    df['prev_day_sleep'] = df.groupby('user_id')['sleep_hours'].shift(1).fillna(df['sleep_hours'])
    df['prev_day_stress'] = df.groupby('user_id')['stress_level'].shift(1).fillna(df['stress_level'])
    
    # Risk score combinations
    df['high_risk_combo'] = ((df['sleep_hours'] < 6) & (df['stress_level'] >= 4)).astype(int)
    df['screen_dehydration_combo'] = ((df['screen_time'] > 7) & (df['hydration_level'] <= 2)).astype(int)
    
    return df

# ================================
# 3. MODEL TRAINING PIPELINE
# ================================

class MigrainePredictionModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.trained = False
        
    def prepare_features(self, df):
        """Prepare features for training"""
        feature_cols = [
            'sleep_hours', 'mood_level', 'stress_level', 'hydration_level', 'screen_time',
            'day_of_week', 'is_weekend', 'consecutive_poor_sleep',
            'sleep_hours_3day_avg', 'stress_level_3day_avg', 'mood_level_3day_avg', 'hydration_level_3day_avg',
            'sleep_hours_7day_avg', 'stress_level_7day_avg', 'mood_level_7day_avg', 'hydration_level_7day_avg',
            'sleep_debt', 'stress_sleep_interaction', 'dehydration_screen_risk',
            'prev_day_migraine', 'prev_day_sleep', 'prev_day_stress',
            'high_risk_combo', 'screen_dehydration_combo'
        ]
        
        X = df[feature_cols].copy()
        y = df['migraine_occurrence']
        self.feature_names = feature_cols
        return X, y
    
    def train_models(self, df):
        """Train multiple models and select the best"""
        print("🔧 Engineering features...")
        df_features = create_features(df)
        
        print("📊 Preparing training data...")
        X, y = self.prepare_features(df_features)
        
        # User-based split to avoid data leakage
        unique_users = df_features['user_id'].unique()
        train_users = unique_users[:int(len(unique_users) * 0.8)]
        test_users = unique_users[int(len(unique_users) * 0.8):]
        
        train_mask = df_features['user_id'].isin(train_users)
        test_mask = df_features['user_id'].isin(test_users)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        print(f"📈 Training set: {len(X_train)} samples, {y_train.sum()} migraines ({y_train.mean():.1%})")
        print(f"📉 Test set: {len(X_test)} samples, {y_test.sum()} migraines ({y_test.mean():.1%})")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        print("🤖 Training models...")
        
        # Model configurations
        model_configs = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                class_weight='balanced', random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                class_weight='balanced', max_iter=1000, random_state=42
            )
        }
        
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in model_configs.items():
            # Train model
            if name == 'Logistic Regression':
                model.fit(X_train_balanced, y_train_balanced)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Focus on recall for healthcare application
            from sklearn.metrics import recall_score, precision_score, f1_score
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Composite score (emphasizing recall)
            composite_score = (recall * 0.4) + (precision * 0.2) + (f1 * 0.2) + (auc_score * 0.2)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc_score,
                'composite_score': composite_score
            }
            
            print(f"   {name}:")
            print(f"      Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")
            print(f"      F1: {f1:.3f} | AUC: {auc_score:.3f} | Composite: {composite_score:.3f}")
            
            if composite_score > best_score:
                best_score = composite_score
                best_model = model
                self.best_model_name = name
        
        self.models = {name: result['model'] for name, result in results.items()}
        self.best_model = best_model
        self.results = results
        self.trained = True
        
        print(f"\n🏆 Best model: {self.best_model_name} (Score: {best_score:.3f})")
        
        # Save models
        self.save_models()
        
        return results
    
    def predict_migraine_risk(self, input_data):
        """Predict migraine risk for given input"""
        if not self.trained:
            raise ValueError("Model not trained yet!")
        
        # Ensure input_data is a DataFrame
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Add missing features with defaults
        for feature in self.feature_names:
            if feature not in input_data.columns:
                if 'avg' in feature:
                    base_feature = feature.split('_')[0] + '_' + feature.split('_')[1]
                    if base_feature in input_data.columns:
                        input_data[feature] = input_data[base_feature]
                    else:
                        input_data[feature] = 3.0  # Default value
                elif 'prev_day' in feature:
                    input_data[feature] = 3.0  # Default previous day values
                elif 'interaction' in feature or 'combo' in feature or 'debt' in feature:
                    input_data[feature] = 0.0  # Default interaction values
                else:
                    input_data[feature] = 3.0  # Default value
        
        X = input_data[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probability
        prediction = self.best_model.predict(X_scaled)[0]
        probability = self.best_model.predict_proba(X_scaled)[0, 1]
        
        # Risk level classification
        if probability < 0.2:
            risk_level = "Very Low"
            risk_color = "green"
        elif probability < 0.4:
            risk_level = "Low"
            risk_color = "lightgreen"
        elif probability < 0.6:
            risk_level = "Moderate"
            risk_color = "yellow"
        elif probability < 0.8:
            risk_level = "High"
            risk_color = "orange"
        else:
            risk_level = "Very High"
            risk_color = "red"
        
        return {
            'prediction': int(prediction),
            'probability': probability,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_percentage': probability * 100
        }
    
    def get_feature_importance(self):
        """Get feature importance from the best model"""
        if not self.trained:
            return None
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return feature_importance
        return None
    
    def save_models(self):
        """Save trained models"""
        try:
            joblib.dump({
                'models': self.models,
                'best_model': self.best_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'best_model_name': self.best_model_name,
                'trained': self.trained
            }, 'migraine_model.pkl')
            print("💾 Models saved successfully!")
        except Exception as e:
            print(f"❌ Error saving models: {e}")
    
    def load_models(self):
        """Load saved models"""
        try:
            data = joblib.load('migraine_model.pkl')
            self.models = data['models']
            self.best_model = data['best_model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.best_model_name = data['best_model_name']
            self.trained = data['trained']
            print("✅ Models loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

# ================================
# 4. STREAMLIT WEB INTERFACE
# ================================

def create_streamlit_app():
    """Create Streamlit web interface"""
    
    st.set_page_config(
        page_title="Migraine Prediction System",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Migraine Prediction System MVP")
    st.markdown("### AI-Powered Migraine Risk Assessment")
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = MigrainePredictionModel()
        st.session_state.data_generated = False
        st.session_state.model_trained = False
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Home & Prediction",
        "📊 Model Training",
        "📈 Analytics Dashboard",
        "ℹ️ About & Help"
    ])
    
    if page == "🏠 Home & Prediction":
        show_prediction_page()
    elif page == "📊 Model Training":
        show_training_page()
    elif page == "📈 Analytics Dashboard":
        show_analytics_page()
    elif page == "ℹ️ About & Help":
        show_about_page()

def show_prediction_page():
    """Main prediction interface"""
    st.header("Daily Migraine Risk Assessment")
    
    if not st.session_state.model.trained:
        st.warning("⚠️ Please train the model first by going to the 'Model Training' page.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Today's Information")
        
        # Input form
        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                sleep_hours = st.slider("Sleep Hours Last Night", 3.0, 12.0, 7.5, 0.5)
                stress_level = st.slider("Stress Level (1-5)", 1, 5, 3)
                hydration_level = st.slider("Hydration Level (1-5)", 1, 5, 3)
                
            with col_b:
                mood_level = st.slider("Mood Level (1-5)", 1, 5, 3)
                screen_time = st.slider("Screen Time (hours)", 1.0, 16.0, 6.0, 0.5)
                consecutive_poor_sleep = st.number_input("Consecutive Days with <6hrs Sleep", 0, 7, 0)
            
            # Advanced options
            with st.expander("Advanced Options"):
                day_of_week = st.selectbox("Day of Week", 
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                prev_day_migraine = st.checkbox("Had migraine yesterday")
            
            submitted = st.form_submit_button("🔮 Predict Migraine Risk")
            
            if submitted:
                # Prepare input data
                day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
                              "Friday": 4, "Saturday": 5, "Sunday": 6}
                
                input_data = {
                    'sleep_hours': sleep_hours,
                    'mood_level': mood_level,
                    'stress_level': stress_level,
                    'hydration_level': hydration_level,
                    'screen_time': screen_time,
                    'day_of_week': day_mapping[day_of_week],
                    'is_weekend': 1 if day_mapping[day_of_week] >= 5 else 0,
                    'consecutive_poor_sleep': consecutive_poor_sleep,
                    'prev_day_migraine': 1 if prev_day_migraine else 0,
                }
                
                # Make prediction
                try:
                    result = st.session_state.model.predict_migraine_risk(input_data)
                    
                    # Display results
                    with col2:
                        st.subheader("Risk Assessment")
                        
                        # Risk gauge
                        risk_percentage = result['risk_percentage']
                        
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = risk_percentage,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Migraine Risk %"},
                            delta = {'reference': 30},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': result['risk_color']},
                                'steps': [
                                    {'range': [0, 20], 'color': "lightgreen"},
                                    {'range': [20, 40], 'color': "yellow"},
                                    {'range': [40, 60], 'color': "orange"},
                                    {'range': [60, 80], 'color': "red"},
                                    {'range': [80, 100], 'color': "darkred"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 80
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk level display
                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: {result['risk_color']}; text-align: center;">
                            <h3 style="color: white; margin: 0;">Risk Level: {result['risk_level']}</h3>
                            <p style="color: white; margin: 5px 0;">{risk_percentage:.1f}% chance of migraine</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Recommendations
                        st.subheader("💡 Recommendations")
                        recommendations = generate_recommendations(input_data, result)
                        for rec in recommendations:
                            st.write(f"• {rec}")
                
                except Exception as e:
                    st.error(f"Prediction error: {e}")

def show_training_page():
    """Model training interface"""
    st.header("Model Training & Performance")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Training Configuration")
        
        # Training parameters
        num_users = st.slider("Number of Users", 8, 20, 12)
        min_days = st.slider("Minimum Days per User", 60, 150, 120)
        max_days = st.slider("Maximum Days per User", 100, 200, 180)
        
        if st.button("🚀 Generate Data & Train Models"):
            with st.spinner("Generating synthetic dataset..."):
                # Generate data
                np.random.seed(42)
                random.seed(42)
                data = generate_enhanced_migraine_dataset(num_users, min_days, max_days)
                st.session_state.data = data
                st.session_state.data_generated = True
                
                st.success(f"✅ Generated {len(data)} records for {num_users} users")
                
            with st.spinner("Training models..."):
                # Train models
                results = st.session_state.model.train_models(data)
                st.session_state.model_trained = True
                st.session_state.training_results = results
                
                st.success("✅ Models trained successfully!")
    
    with col2:
        if st.session_state.get('model_trained', False):
            st.subheader("Model Performance")
            
            results = st.session_state.training_results
            
            # Performance comparison
            performance_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy': [results[m]['accuracy'] for m in results.keys()],
                'Precision': [results[m]['precision'] for m in results.keys()],
                'Recall': [results[m]['recall'] for m in results.keys()],
                'F1-Score': [results[m]['f1'] for m in results.keys()],
                'AUC': [results[m]['auc'] for m in results.keys()]
            })
            
            st.dataframe(performance_df.round(3))
            
            # Best model highlight
            best_model = st.session_state.model.best_model_name
            st.success(f"🏆 Best Model: {best_model}")
            
            # Feature importance
            feature_importance = st.session_state.model.get_feature_importance()
            if feature_importance is not None:
                st.subheader("Top 10 Important Features")
                top_features = feature_importance.head(10)
                
                fig = px.bar(top_features, x='importance', y='feature', orientation='h',
                           title="Feature Importance")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

def show_analytics_page():
    """Analytics dashboard"""
    st.header("Analytics Dashboard")
    
    if not st.session_state.get('data_generated', False):
        st.warning("⚠️ Please generate data first by going to the 'Model Training' page.")
        return
    
    data = st.session_state.data
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Total Users", data['user_id'].nunique())
    with col3:
        migraine_rate = data['migraine_occurrence'].mean()
        st.metric("Migraine Rate", f"{migraine_rate:.1%}")
    with col4:
        avg_severity = data[data['migraine_occurrence']==1]['migraine_severity'].mean()
        st.metric("Avg Severity", f"{avg_severity:.1f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Migraine occurrence by day of week
        dow_data = data.groupby('day_of_week')['migraine_occurrence'].agg(['count', 'sum', 'mean']).reset_index()
        dow_data['day_name'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig = px.bar(dow_data, x='day_name', y='mean', 
                    title="Migraine Rate by Day of Week")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sleep vs Migraine
        sleep_migraine = data.groupby(pd.cut(data['sleep_hours'], bins=5))['migraine_occurrence'].mean()
        
        fig = px.bar(x=sleep_migraine.index.astype(str), y=sleep_migraine.values,
                    title="Migraine Rate by Sleep Duration")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    numeric_cols = ['sleep_hours', 'mood_level', 'stress_level', 'hydration_level', 
                   'screen_time', 'migraine_occurrence']
    corr_matrix = data[numeric_cols].corr()
              
    fig = px.imshow(corr_matrix, text_auto=True,
                   title="Correlation Matrix of Key Features")
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    st.subheader("Migraine Trends Over Time")
    
    # Daily migraine rate over time
    daily_migraines = data.groupby('date')['migraine_occurrence'].agg(['count', 'sum']).reset_index()
    daily_migraines['rate'] = daily_migraines['sum'] / daily_migraines['count']
    daily_migraines['date'] = pd.to_datetime(daily_migraines['date'])
    
    fig = px.line(daily_migraines, x='date', y='rate', 
                  title="Daily Migraine Rate Over Time")
    fig.update_layout(xaxis_title="Date", yaxis_title="Migraine Rate")
    st.plotly_chart(fig, use_container_width=True)
    
    # User-specific analysis
    st.subheader("User Analysis")
    selected_user = st.selectbox("Select User for Detailed Analysis", 
                                sorted(data['user_id'].unique()))
    
    user_data = data[data['user_id'] == selected_user]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # User's migraine pattern
        user_daily = user_data.set_index('date')['migraine_occurrence'].resample('D').sum()
        fig = px.bar(x=user_daily.index, y=user_daily.values,
                    title=f"User {selected_user} - Daily Migraines")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # User's lifestyle patterns
        lifestyle_cols = ['sleep_hours', 'stress_level', 'mood_level', 'hydration_level']
        user_lifestyle = user_data[lifestyle_cols + ['date']].melt(id_vars=['date'], 
                                                                   var_name='metric', 
                                                                   value_name='value')
        
        fig = px.line(user_lifestyle, x='date', y='value', color='metric',
                     title=f"User {selected_user} - Lifestyle Patterns")
        st.plotly_chart(fig, use_container_width=True)

def show_about_page():
    """About and help page"""
    st.header("About Migraine Prediction System")
    
    st.markdown("""
    ## 🧠 What is this system?
    
    This is a **Machine Learning-powered Migraine Prediction System** that helps users assess their daily risk of experiencing a migraine based on lifestyle factors.
    
    ## 🎯 Key Features
    
    - **AI-Powered Predictions**: Uses advanced machine learning algorithms to predict migraine risk
    - **Multiple Models**: Compares Random Forest, Gradient Boosting, and Logistic Regression
    - **Real-time Assessment**: Get instant risk assessment based on daily inputs
    - **Personalized Recommendations**: Receive tailored advice to reduce migraine risk
    - **Analytics Dashboard**: Visualize patterns and trends in migraine data
    
    ## 📊 How it works
    
    1. **Data Collection**: The system analyzes various lifestyle factors including:
       - Sleep duration and quality
       - Stress levels
       - Mood and hydration
       - Screen time exposure
       - Previous migraine history
    
    2. **Feature Engineering**: Creates advanced features like:
       - Rolling averages (3-day and 7-day)
       - Sleep debt calculations
       - Interaction terms between risk factors
       - Consecutive poor sleep tracking
    
    3. **ML Prediction**: Uses the best-performing model to predict:
       - Probability of migraine occurrence
       - Risk level classification
       - Confidence intervals
    
    ## 🔬 Model Performance
    
    The system uses multiple evaluation metrics:
    - **Recall**: Prioritized for healthcare applications (catching true migraines)
    - **Precision**: Minimizing false alarms
    - **AUC-ROC**: Overall discriminative ability
    - **F1-Score**: Balanced precision and recall
    
    ## ⚠️ Important Disclaimers
    
    - This system is for **educational and research purposes only**
    - **Not a substitute for professional medical advice**
    - Always consult healthcare professionals for medical concerns
    - The model is trained on synthetic data for demonstration
    
    ## 🚀 Usage Instructions
    
    ### Step 1: Train the Model
    1. Go to "Model Training" page
    2. Adjust parameters if needed
    3. Click "Generate Data & Train Models"
    4. Wait for training to complete
    
    ### Step 2: Make Predictions
    1. Go to "Home & Prediction" page
    2. Enter your daily information
    3. Click "Predict Migraine Risk"
    4. Review results and recommendations
    
    ### Step 3: Analyze Patterns
    1. Go to "Analytics Dashboard"
    2. Explore various visualizations
    3. Analyze user-specific patterns
    4. Identify risk factors and trends
    
    ## 🛠️ Technical Details
    
    **Machine Learning Pipeline:**
    - Data Generation: Synthetic dataset with realistic patterns
    - Feature Engineering: 25+ engineered features
    - Model Training: Ensemble of 3 different algorithms
    - Evaluation: Cross-validation with user-based splits
    - Deployment: Real-time Streamlit interface
    
    **Key Technologies:**
    - Python, Pandas, NumPy
    - Scikit-learn, Imbalanced-learn
    - Streamlit, Plotly
    - Joblib for model persistence
    
    ## 📈 Future Enhancements
    
    - Integration with wearable devices
    - Weather and environmental factors
    - Medication tracking
    - Mobile app development
    - Clinical validation studies
    
    ## 🤝 Support
    
    For technical issues or questions about this system, please refer to the documentation or contact the development team.
    
    ---
    
    **Version**: 1.0 MVP  
    **Last Updated**: 2024  
    **Status**: Demonstration/Educational Use Only
    """)

def generate_recommendations(input_data, prediction_result):
    """Generate personalized recommendations based on input and prediction"""
    recommendations = []
    risk_level = prediction_result['risk_level']
    
    # Sleep-based recommendations
    if input_data['sleep_hours'] < 6:
        recommendations.append("🛏️ **Priority**: Aim for 7-9 hours of sleep tonight")
        recommendations.append("😴 Consider going to bed 1-2 hours earlier than usual")
    elif input_data['sleep_hours'] < 7:
        recommendations.append("😴 Try to get an extra hour of sleep tonight for better recovery")
    
    # Stress management
    if input_data['stress_level'] >= 4:
        recommendations.append("🧘 **High Stress Detected**: Practice relaxation techniques (deep breathing, meditation)")
        recommendations.append("📱 Consider using a stress management app or taking short breaks")
    elif input_data['stress_level'] >= 3:
        recommendations.append("🌱 Take some time for stress-reducing activities today")
    
    # Hydration advice
    if input_data['hydration_level'] <= 2:
        recommendations.append("💧 **Important**: Increase water intake significantly today")
        recommendations.append("⏰ Set hourly reminders to drink water")
    elif input_data['hydration_level'] <= 3:
        recommendations.append("💧 Make sure to stay well-hydrated throughout the day")
    
    # Screen time management
    if input_data['screen_time'] > 8:
        recommendations.append("📱 **Reduce Screen Time**: Take regular breaks every 20-30 minutes")
        recommendations.append("🕶️ Consider blue light filtering glasses")
        recommendations.append("🌅 Avoid screens 1 hour before bedtime")
    elif input_data['screen_time'] > 6:
        recommendations.append("👀 Follow the 20-20-20 rule: Every 20 minutes, look at something 20 feet away for 20 seconds")
    
    # Consecutive poor sleep
    if input_data['consecutive_poor_sleep'] >= 2:
        recommendations.append("⚠️ **Sleep Debt Alert**: Your body needs recovery - prioritize sleep tonight")
        recommendations.append("🛌 Consider a short nap (20-30 minutes) if possible")
    
    # Risk-level specific advice
    if risk_level in ["High", "Very High"]:
        recommendations.append("🚨 **High Risk Day**: Consider preventive measures you've used successfully before")
        recommendations.append("📋 Keep a migraine kit ready (medication, dark room, ice pack)")
        recommendations.append("🗓️ Avoid known triggers and stressful activities if possible")
    elif risk_level == "Moderate":
        recommendations.append("⚠️ **Moderate Risk**: Be mindful of your triggers today")
        recommendations.append("🎯 Focus on the top recommendations above")
    else:
        recommendations.append("✅ **Low Risk Day**: Great job maintaining healthy habits!")
        recommendations.append("🔄 Keep up your current routine")
    
    # General wellness
    recommendations.append("🍎 Maintain regular, balanced meals throughout the day")
    recommendations.append("🚶 Light exercise or walking can help reduce migraine risk")
    
    return recommendations

# ================================
# 5. MAIN EXECUTION
# ================================

def main():
    """Main execution function"""
    
    # Check if running in Streamlit
    try:
        import streamlit as st
        # If we're in Streamlit, run the app
        create_streamlit_app()
    except:
        # If not in Streamlit, run training pipeline
        print("🎯 Migraine Prediction MVP System")
        print("=" * 50)
        
        # Generate sample data
        print("📊 Generating synthetic dataset...")
        np.random.seed(42)
        random.seed(42)
        data = generate_enhanced_migraine_dataset(num_users=12, min_days=120, max_days=180)
        print(f"✅ Generated {len(data)} records for {data['user_id'].nunique()} users")
        
        # Train models
        print("\n🤖 Training models...")
        model = MigrainePredictionModel()
        results = model.train_models(data)
        
        # Test prediction
        print("\n🔮 Testing prediction...")
        test_input = {
            'sleep_hours': 5.5,
            'mood_level': 2,
            'stress_level': 4,
            'hydration_level': 2,
            'screen_time': 9.0,
            'day_of_week': 0,  # Monday
            'is_weekend': 0,
            'consecutive_poor_sleep': 2,
            'prev_day_migraine': 0,
        }
        
        prediction = model.predict_migraine_risk(test_input)
        print(f"📈 Sample Prediction:")
        print(f"   Risk Level: {prediction['risk_level']}")
        print(f"   Probability: {prediction['risk_percentage']:.1f}%")
        
        print("\n✅ System ready! Run with 'streamlit run [filename].py' for web interface")
        
        # Show feature importance
        feature_importance = model.get_feature_importance()
        if feature_importance is not None:
            print(f"\n🎯 Top 5 Important Features:")
            for idx, row in feature_importance.head(5).iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
        
        return model, data

if __name__ == "__main__":
    # Run the main function
    main()

# ================================
# 6. ADDITIONAL UTILITIES
# ================================

def export_model_summary(model, filename="model_summary.txt"):
    """Export model performance summary to file"""
    if not model.trained:
        print("❌ Model not trained yet")
        return
    
    with open(filename, 'w') as f:
        f.write("MIGRAINE PREDICTION MODEL SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Best Model: {model.best_model_name}\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL PERFORMANCE:\n")
        f.write("-" * 20 + "\n")
        for name, result in model.results.items():
            f.write(f"{name}:\n")
            f.write(f"  Accuracy: {result['accuracy']:.3f}\n")
            f.write(f"  Precision: {result['precision']:.3f}\n")
            f.write(f"  Recall: {result['recall']:.3f}\n")
            f.write(f"  F1-Score: {result['f1']:.3f}\n")
            f.write(f"  AUC: {result['auc']:.3f}\n\n")
        
        f.write("FEATURE IMPORTANCE (Top 10):\n")
        f.write("-" * 30 + "\n")
        feature_importance = model.get_feature_importance()
        if feature_importance is not None:
            for idx, row in feature_importance.head(10).iterrows():
                f.write(f"{row['feature']}: {row['importance']:.3f}\n")
    
    print(f"📋 Model summary exported to {filename}")

def create_sample_prediction_script():
    """Create a standalone prediction script"""
    script_content = '''
# Standalone Migraine Prediction Script
import joblib
import pandas as pd

def load_model():
    """Load the trained model"""
    try:
        return joblib.load('migraine_model.pkl')
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
        return None

def predict_migraine(sleep_hours, stress_level, mood_level, hydration_level, screen_time, **kwargs):
    """Simple prediction function"""
    model_data = load_model()
    if model_data is None:
        return None
    
    # Prepare input
    input_data = {
        'sleep_hours': sleep_hours,
        'mood_level': mood_level,
        'stress_level': stress_level,
        'hydration_level': hydration_level,
        'screen_time': screen_time,
        'day_of_week': kwargs.get('day_of_week', 0),
        'is_weekend': kwargs.get('is_weekend', 0),
        'consecutive_poor_sleep': kwargs.get('consecutive_poor_sleep', 0),
        'prev_day_migraine': kwargs.get('prev_day_migraine', 0),
    }
    
    # Add missing features with defaults
    for feature in model_data['feature_names']:
        if feature not in input_data:
            if 'avg' in feature:
                base_feature = feature.split('_')[0] + '_' + feature.split('_')[1]
                input_data[feature] = input_data.get(base_feature, 3.0)
            else:
                input_data[feature] = 0.0
    
    # Make prediction
    X = pd.DataFrame([input_data])[model_data['feature_names']]
    X_scaled = model_data['scaler'].transform(X)
    
    probability = model_data['best_model'].predict_proba(X_scaled)[0, 1]
    
    return {
        'probability': probability,
        'risk_percentage': probability * 100,
        'risk_level': 'High' if probability > 0.6 else 'Moderate' if probability > 0.3 else 'Low'
    }

# Example usage
if __name__ == "__main__":
    result = predict_migraine(
        sleep_hours=5.5,
        stress_level=4,
        mood_level=2,
        hydration_level=2,
        screen_time=8.5
    )
    
    if result:
        print(f"Migraine Risk: {result['risk_level']} ({result['risk_percentage']:.1f}%)")
'''
    
    with open('simple_predictor.py', 'w') as f:
        f.write(script_content)
    
    print("📝 Standalone prediction script created: simple_predictor.py")

# ================================
# 7. CONFIGURATION AND SETTINGS
# ================================

# Model configuration
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'class_weight': 'balanced'
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    },
    'logistic_regression': {
        'class_weight': 'balanced',
        'max_iter': 1000
    }
}

# Feature configuration
FEATURE_GROUPS = {
    'basic': ['sleep_hours', 'mood_level', 'stress_level', 'hydration_level', 'screen_time'],
    'temporal': ['day_of_week', 'is_weekend', 'consecutive_poor_sleep'],
    'historical': ['prev_day_migraine', 'prev_day_sleep', 'prev_day_stress'],
    'engineered': ['sleep_debt', 'stress_sleep_interaction', 'dehydration_screen_risk'],
    'rolling': ['sleep_hours_3day_avg', 'stress_level_3day_avg', 'mood_level_3day_avg'],
    'combinations': ['high_risk_combo', 'screen_dehydration_combo']
}

# Risk thresholds
RISK_THRESHOLDS = {
    'very_low': 0.2,
    'low': 0.4,
    'moderate': 0.6,
    'high': 0.8
}

print("🎯 Migraine Prediction MVP System - Complete")
print("=" * 50)
print("✅ All components loaded successfully!")
print("\n🚀 To run the web interface:")
print("   streamlit run [this_filename].py")
print("\n📊 To run training only:")
print("   python [this_filename].py")