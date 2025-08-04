from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X = iris.data      # Features (sepal/petal length and width)
y = iris.target    # Labels (0=setosa, 1=versicolor, 2=virginica)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy * 100:.2f}%")

# Predict a new sample
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example input
prediction = model.predict(sample)
print(f"Predicted class: {iris.target_names[prediction[0]]}")

# Module 1 code here
