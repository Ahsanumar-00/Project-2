
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


from sklearn.datasets import load_iris

iris = load_iris()


df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)


df["target"] = iris.target


print("==========================================")
print("       DATA CLASSIFICATION USING AI")
print("==========================================")

print("\nFirst 5 rows of the dataset:")
print(df.head())


print("\nDataset Information:")
print("------------------------------------------")

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nTarget classes:")
print(iris.target_names)


X = df.drop("target", axis=1)
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nData Split:")
print("------------------------------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


model = KNeighborsClassifier(n_neighbors=3)


model.fit(X_train, y_train)

print("\nModel training completed successfully!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print("------------------------------------------")
print(f"Accuracy: {accuracy * 100:.2f}%")


print("\nClassification Report:")
print("------------------------------------------")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


print("\nTesting the model with new data:")
print("------------------------------------------")


new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

predicted_class = iris.target_names[prediction[0]]

print("New flower measurements:", new_flower)
print("Predicted class:", predicted_class)


print("\n==========================================")
print("       PROJECT COMPLETED SUCCESSFULLY")
print("==========================================")