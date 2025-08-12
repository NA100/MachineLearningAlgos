from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Sample data
data = {
    'Fruit': ['Apple', 'Banana', 'Orange', 'Apple', 'Orange', 'Banana']
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Create LabelEncoder instance
encoder = LabelEncoder()

# Fit and transform the column
df['Fruit_Encoded'] = encoder.fit_transform(df['Fruit'])

print("\nAfter Label Encoding:")
print(df)

# To see the mapping
label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("\nLabel Mapping:", label_mapping)
