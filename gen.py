import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_migraine_dataset(num_users=10, min_days=60, max_days=180):
    """
    Generate a realistic migraine tracking dataset for multiple users
    
    Parameters:
    - num_users: Number of synthetic users to generate data for
    - min_days: Minimum number of days per user
    - max_days: Maximum number of days per user
    
    Returns:
    - DataFrame with migraine tracking data
    """
    
    all_data = []
    
    for user_id in range(1, num_users + 1):
        # Random number of days for this user
        num_days = random.randint(min_days, max_days)
        
        # Generate base date range
        start_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 100))
        dates = [start_date + timedelta(days=i) for i in range(num_days)]
        
        # User-specific baseline characteristics
        user_sleep_baseline = np.random.normal(7.5, 1.0)  # Individual sleep pattern
        user_stress_baseline = np.random.uniform(1.5, 4.5)  # Individual stress level (1-5 scale)
        user_migraine_proneness = np.random.beta(2, 8)      # Some users more prone to migraines
        
        for day_idx, date in enumerate(dates):
            # Generate correlated features with realistic patterns
            
            # Sleep hours (with weekend patterns)
            is_weekend = date.weekday() >= 5
            sleep_base = user_sleep_baseline + (0.5 if is_weekend else 0)
            sleep_hours = max(3.0, min(12.0, np.random.normal(sleep_base, 0.8)))
            
            # Stress level (higher on weekdays, with some personal variation)
            stress_base = user_stress_baseline + (-1.5 if is_weekend else 0.5)
            stress_level = max(0, min(10, np.random.normal(stress_base, 1.2)))
            
            # Stress level (1-5 scale, higher on weekdays)
            stress_base = (user_stress_baseline / 2) + (-0.5 if is_weekend else 0.3)
            stress_level = max(1, min(5, int(np.random.normal(stress_base, 0.7))))
            
            # Mood level (1-5 scale, inversely related to stress, positively to sleep)
            mood_base = 4.5 - (stress_level * 0.4) + ((sleep_hours - 6) * 0.3)
            mood_level = max(1, min(5, int(np.random.normal(mood_base, 0.6))))
            
            # Hydration level (1-5 scale, with some randomness)
            hydration_level = max(1, min(5, int(np.random.normal(3.2, 0.9))))
            
            # Screen time (higher on weekends and when stressed)
            screen_base = 5 + (1 if is_weekend else 0) + (stress_level * 0.3)
            screen_time = max(1.0, min(16.0, np.random.normal(screen_base, 1.5)))
            
            # Migraine occurrence logic - reflecting key observations
            migraine_risk = user_migraine_proneness
            
            # OBSERVATION 1: Less than 6 hours sleep + high stress = higher migraine chances
            if sleep_hours < 6 and stress_level >= 4:
                migraine_risk += 0.6  # Strong combined effect
            elif sleep_hours < 6:
                migraine_risk += 0.3  # Sleep deprivation alone
            elif stress_level >= 4:
                migraine_risk += 0.25  # High stress alone
                
            # OBSERVATION 2: Higher screen time + low hydration = noticeable impact
            if screen_time > 7 and hydration_level <= 2:
                migraine_risk += 0.4  # Combined screen time and dehydration
            elif screen_time > 8:
                migraine_risk += 0.15  # High screen time alone
            elif hydration_level <= 2:
                migraine_risk += 0.3   # Severe dehydration
            elif hydration_level <= 3:
                migraine_risk += 0.15  # Mild dehydration
                
            # Additional moderate risk factors
            if stress_level == 5:  # Maximum stress
                migraine_risk += 0.2
                
            # OBSERVATION 3: Consecutive poor sleep days increase risk
            consecutive_poor_sleep = 0
            if day_idx >= 1:
                # Check previous day's sleep
                if len(all_data) > 0 and all_data[-1]['sleep_hours'] < 6:
                    consecutive_poor_sleep += 1
                    if sleep_hours < 6:  # Current day also poor sleep
                        migraine_risk += 0.4  # Significant increase for consecutive poor sleep
                    else:
                        migraine_risk += 0.2  # Previous day effect
                        
            if day_idx >= 2:
                # Check two days ago for 3-day pattern
                if (len(all_data) >= 2 and 
                    all_data[-2]['sleep_hours'] < 6 and 
                    all_data[-1]['sleep_hours'] < 6 and 
                    sleep_hours < 6):
                    migraine_risk += 0.3  # Additional risk for 3+ consecutive poor sleep days
                
            # Add some day-of-week patterns (Monday stress, weekend changes)
            if date.weekday() == 0:  # Monday
                migraine_risk += 0.05
                
            # Previous day migraine increases next day risk slightly
            if day_idx > 0 and all_data[-1]['migraine_occurrence'] == 1:
                migraine_risk += 0.15
            
            # Cap the risk
            migraine_risk = min(0.8, migraine_risk)
            
            # Determine migraine occurrence
            migraine_occurrence = 1 if np.random.random() < migraine_risk else 0
            
            # Migraine severity (only if migraine occurs)
            if migraine_occurrence == 1:
                severity_base = 1.5
                
                # Severity increases with poor conditions (adjusted for 1-5 scales)
                if hydration_level <= 2:
                    severity_base += 0.8
                elif hydration_level <= 3:
                    severity_base += 0.3
                    
                if sleep_hours < 5:
                    severity_base += 0.6
                elif sleep_hours < 6:
                    severity_base += 0.3
                    
                if stress_level >= 4:
                    severity_base += 0.4
                elif stress_level >= 3:
                    severity_base += 0.2
                
                # Consecutive poor sleep increases severity
                if consecutive_poor_sleep > 0:
                    severity_base += 0.3
                
                migraine_severity = max(1, min(3, int(np.random.normal(severity_base, 0.5))))
            else:
                migraine_severity = 0
            
            # Round values appropriately
            data_point = {
                'user_id': user_id,
                'date': date.strftime('%Y-%m-%d'),
                'sleep_hours': round(sleep_hours, 1),
                'mood_level': mood_level,
                'stress_level': stress_level,
                'hydration_level': hydration_level,
                'screen_time': round(screen_time, 1),
                'migraine_occurrence': migraine_occurrence,
                'migraine_severity': migraine_severity
            }
            
            all_data.append(data_point)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by user and date
    df = df.sort_values(['user_id', 'date']).reset_index(drop=True)
    
    return df

# Generate the dataset
np.random.seed(42)  # For reproducible results
random.seed(42)

# Create dataset with 8 users, 90-150 days each
migraine_data = generate_migraine_dataset(num_users=100, min_days=90, max_days=150)

# Display basic information about the dataset
print("Migraine Dataset Generated!")
print(f"Total records: {len(migraine_data)}")
print(f"Number of users: {migraine_data['user_id'].nunique()}")
print(f"Date range: {migraine_data['date'].min()} to {migraine_data['date'].max()}")
print(f"Migraine occurrence rate: {migraine_data['migraine_occurrence'].mean():.2%}")

print("\nDataset sample:")
print(migraine_data.head(10))

print("\nDataset statistics:")
print(migraine_data.describe())

print("\nMigraine severity distribution (when migraines occur):")
migraine_days = migraine_data[migraine_data['migraine_occurrence'] == 1]
print(migraine_days['migraine_severity'].value_counts().sort_index())

print("\nCorrelation with migraine occurrence:")
correlations = migraine_data.select_dtypes(include=[np.number]).corr()['migraine_occurrence'].sort_values(ascending=False)
print(correlations[correlations.index != 'migraine_occurrence'])

# Save to CSV
migraine_data.to_csv('migraine_dataset.csv', index=False)
print("\nDataset saved as 'migraine_dataset.csv'")

# Example: Analyze patterns for one user
user_1_data = migraine_data[migraine_data['user_id'] == 1].copy()
print(f"\nUser 1 Analysis ({len(user_1_data)} days):")
print(f"Migraine days: {user_1_data['migraine_occurrence'].sum()}")
print(f"Average sleep on migraine days: {user_1_data[user_1_data['migraine_occurrence']==1]['sleep_hours'].mean():.1f} hours")
print(f"Average sleep on non-migraine days: {user_1_data[user_1_data['migraine_occurrence']==0]['sleep_hours'].mean():.1f} hours")
print(f"Average stress on migraine days: {user_1_data[user_1_data['migraine_occurrence']==1]['stress_level'].mean():.1f}")
print(f"Average stress on non-migraine days: {user_1_data[user_1_data['migraine_occurrence']==0]['stress_level'].mean():.1f}")