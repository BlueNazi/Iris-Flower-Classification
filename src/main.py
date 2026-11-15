from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def main():
    # Load the sample Iris dataset provided by scikit-learn.
    iris = load_iris()
    features = iris.data
    labels = iris.target

    # Keep some data aside to evaluate the model.
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    # Train a simple classification model.
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on data it did not train on.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy:.2%}")

    # Predict the species for a sample flower:
    # [sepal length, sepal width, petal length, petal width] in cm.
    sample = [[5.1, 3.5, 1.4, 0.2]]
    predicted_class = model.predict(sample)[0]
    predicted_species = iris.target_names[predicted_class]

    print(f"Predicted species for the sample flower: {predicted_species}")


if __name__ == "__main__":
    main()
