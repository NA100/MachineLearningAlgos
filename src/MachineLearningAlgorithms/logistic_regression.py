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


"""
🧠 What is LogisticRegression?
Despite the name, Logistic Regression is used for classification, not regression.
It is a supervised learning algorithm that predicts discrete categories — for example:

Is the email spam or not?
Is the flower Setosa, Versicolor, or Virginica?

⚙️ How does it work?
Logistic regression calculates a probability that a given input belongs to a certain class, using a mathematical function called the sigmoid (or softmax).
In binary classification:

P(class=1) = 1 / (1 + e^-(w·x + b))
In multi-class classification (like Iris with 3 flower types), it uses softmax to output a probability for each class.
Then it chooses the class with the highest probability.

🧪 Example:
Let’s say you input:
[5.1, 3.5, 1.4, 0.2]  # Features for one flower
The model computes probabilities like:
[0.98, 0.01, 0.01]  # 98% chance it's Setosa
It predicts the flower is Setosa (class 0).
🧰 In Code:
You create and train a logistic regression model like this:
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)  # Train the model
predictions = model.predict(X_test)  # Predict on new data

✅ When to Use Logistic Regression?
When your target is categorical
For binary or multi-class classification
When you want a model that is fast, interpretable, and works well for linear decision boundaries
Would you like a visual explanation or a comparison with other algorithms like decision trees or neural networks?
"""
