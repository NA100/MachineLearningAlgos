from sklearn.model_selection import train_test_split
import pandas as pd
"""
Key points:
test_size → proportion of test data (0.2 means 20% test, 80% train).
random_state → ensures the same split every time (good for reproducibility).
stratify=y → keeps class proportions consistent between training and test sets (important for classification problems).
"""
# Sample data
data = {
    'Height': [1.6, 1.65, 1.8, 1.75, 1.7, 1.9],
    'Weight': [60, 65, 80, 75, 68, 85],
    'Sport': ['Tennis', 'Soccer', 'Basketball', 'Tennis', 'Soccer', 'Basketball']
}
df = pd.DataFrame(data)

# Features (X) and target (y)
X = df[['Height', 'Weight']]   # independent variables
y = df['Sport']                # dependent variable

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,     # 20% test data
    random_state=42,   # for reproducibility
    stratify=y         # optional: keeps class distribution the same in train/test
)

print("Training Features:\n", X_train)
print("\nTest Features:\n", X_test)
print("\nTraining Labels:\n", y_train)
print("\nTest Labels:\n", y_test)
