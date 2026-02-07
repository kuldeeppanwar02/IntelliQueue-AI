import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# 1. Folders banana taaki Error na aaye
folders = ['data', 'models']
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 2. Synthetic Data Generate karna
print("⏳ Generating Data...")
data_size = 2000
data = {
    'hour_of_day': np.random.randint(9, 22, data_size), 
    'day_of_week': np.random.randint(0, 7, data_size), 
    'current_staff': np.random.randint(1, 6, data_size),
    'current_crowd': np.random.randint(2, 80, data_size),
}
df = pd.DataFrame(data)
df['wait_time_mins'] = (df['current_crowd'] / df['current_staff'] * 4) + np.random.randint(2, 12, data_size)
df.to_csv('data/queue_data.csv', index=False)

# 3. Model Train karna
print("🧠 Training Model...")
X = df[['hour_of_day', 'day_of_week', 'current_staff', 'current_crowd']]
y = df['wait_time_mins']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Model Save karna
joblib.dump(model, 'models/queue_model.pkl')

print("✅ SUCCESS: Data 'data/' mein hai aur Model 'models/' mein save ho gaya hai!")