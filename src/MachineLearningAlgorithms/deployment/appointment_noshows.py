# Step 1: Train a classification model to predict appointment no-shows

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1. Load Dataset (replace with actual path if needed)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/no-show-appointments.csv"
df = pd.read_csv(url)

# 2. Preprocess
df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['WaitingDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

# Select relevant features
features = ['Age', 'Gender', 'Scholarship', 'Diabetes', 'Alcoholism',
            'Hypertension', 'SMS_received', 'WaitingDays']
X = df[features]
y = df['No-show']

# Drop any rows with missing values
X = X.dropna()
y = y[X.index]

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# 6. Save model
joblib.dump(model, 'no_show_model.pkl')
print("✅ Model saved as 'no_show_model.pkl'")
