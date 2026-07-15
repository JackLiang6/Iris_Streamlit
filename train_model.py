from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Define models to train
models = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "svm": SVC(probability=True, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "tree": DecisionTreeClassifier(random_state=42),
    "logit": LogisticRegression(max_iter=1000, random_state=42),
}

for name, model in models.items():
    # 3. Train models
    model.fit(X_train, y_train)

    # 4. Save models
    filename = f"{name}_model.pkl"
    with open(filename, "wb") as f:
        pickle.dump({"model": model, "classes": iris.target_names}, f)

    print(f"{name} model saved to {filename}")
