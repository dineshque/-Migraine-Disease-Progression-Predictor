# Migraine Disease Progression Predictor

Machine learning application that predicts migraine episodes using temporal patterns and disease progression modeling. Built with Streamlit.

## Features

- **Disease Progression Modeling**: Temporal sequence analysis for accurate predictions
- **Multiple ML Models**: Random Forest, Gradient Boosting, Neural Networks
- **Personal Insights**: Individual risk profiles and recommendations
- **Early Warning System**: Detects high-risk periods 1-3 days in advance

## Quick Start

```bash
git clone https://github.com/dineshque/-Migraine-Disease-Progression-Predictor.git
cd -Migraine-Disease-Progression-Predictor
streamlit run app.py
```

## Data Format

CSV with columns: `user_id`, `date`, `sleep_hours`, `stress_level`, `screen_time_hours`, `hydration_glasses`, `exercise_minutes`, `has_migraine`

## Performance

- **F1-Score**: 0.75+
- **ROC-AUC**: 0.80+
- **Early Warning**: 65% accuracy for 2-day predictions

## Key Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
scikit-learn>=1.3.0
plotly>=5.15.0
```

## Usage

1. Upload migraine tracking data or use sample data
2. Analyze temporal patterns and risk factors
3. Train and compare ML models
4. Generate predictions and personalized insights

Open for Collaboration