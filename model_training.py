import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load sample data
iris = load_iris()
X, y = iris.data, iris.target

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save the model
joblib.dump(model, "random_forest.pkl")
print("Model saved as random_forest.pkl")
