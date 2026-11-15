# Iris Flower Classification

A beginner-friendly machine learning project built with Python and scikit-learn.

## What this project does
This project trains a Decision Tree model to classify Iris flowers into one of three species using four measurements:
- Sepal length
- Sepal width
- Petal length
- Petal width

It uses the Iris dataset included with scikit-learn, so no separate dataset download is needed.

## Requirements
- Python 3.9 or newer
- pip

## Setup

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

The program prints the model's test accuracy and predicts the species for one example flower.

## Skills demonstrated
- Python
- Basic machine learning workflow
- Loading a dataset
- Splitting data into training and test sets
- Training and evaluating a classification model
- Making a prediction with a trained model

## Note
This is an educational starter project. The reported accuracy is based on one train/test split and is not a guarantee of performance on every dataset.
