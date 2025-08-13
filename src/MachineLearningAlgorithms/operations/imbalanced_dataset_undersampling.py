import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv("creditcard.csv")

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split (keep stratified distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Before undersampling:", y_train.value_counts())

# 2. Apply undersampling
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

print("After undersampling:", y_train_res.value_counts())

# 3. Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_res, y_train_res)

# 4. Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
