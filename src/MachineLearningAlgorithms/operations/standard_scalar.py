from sklearn.preprocessing import StandardScaler
import pandas as pd
"""
StandardScaler (from scikit-learn) — it’s a preprocessing tool that
 standardizes your features so they have:
Mean = 0
Standard deviation = 1
In other words, it transforms each feature so it’s centered and scaled,
making the data easier for many machine learning algorithms to work with (especially those sensitive to scale, like SVM, logistic regression, and KNN).
"""
# Sample data
data = {
    'Height': [1.6, 1.65, 1.8, 1.75],
    'Weight': [60, 65, 80, 75]
}
df = pd.DataFrame(data)
print("Original Data:")
print(df)

# Create scaler
scaler = StandardScaler()

# Fit and transform
scaled_data = scaler.fit_transform(df)

# Create new DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
print("\nAfter Standard Scaling:")
print(scaled_df)
