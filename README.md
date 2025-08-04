# Machine Learning Algorithms 

1.   Iris Flower Classification using Logistic Regression 
🌸 What Is Being Done?
We are training a machine learning model to automatically classify iris flowers into one of three species based on their physical measurements.
🧾 The Problem:
Given these 4 features of a flower:
Sepal length (cm)
Sepal width (cm)
Petal length (cm)
Petal width (cm)
➡️ Can we predict which species it is?
0 = Setosa
1 = Versicolor
2 = Virginica
This is a multi-class classification problem (3 possible outcomes).
🤖 What the Code Is Doing:
✅ Step 1: Load the dataset
iris = load_iris()
You get:
iris.data → The flower measurements (features)
iris.target → The flower species (labels)
✅ Step 2: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(...)
We divide the data into:
Training set → used to teach the model
Test set → used to evaluate the model
✅ Step 3: Train the model
model.fit(X_train, y_train)
We use logistic regression (a common classification algorithm). The model learns patterns between the flower measurements and their species.
✅ Step 4: Predict on new data
y_pred = model.predict(X_test)
The model makes predictions on unseen data (the test set).
✅ Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
This tells you how often the model was correct.
✅ Step 6: Use it on new data
sample = [[5.1, 3.5, 1.4, 0.2]]
model.predict(sample)

You can now input new flower measurements, and the model will predict its species.
🔍 Why It’s Useful:
This small project teaches you:
How to load data
Preprocess it
Train a model
Evaluate performance
Make real predictions

Let me know if you'd like a visual or diagram to go with this — or if you'd like to move to another example like image or text classification!



