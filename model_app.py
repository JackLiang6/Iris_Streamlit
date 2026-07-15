import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd

st.set_page_config(page_title="Iris Flower Classifier", page_icon="images/iris_icon.png")

# Models to load: (tab label, filename)
MODEL_FILES = {
    "Random Forest": "random_forest_model.pkl",
    "SVM": "svm_model.pkl",
    "KNN": "knn_model.pkl",
    "Decision Tree": "tree_model.pkl",
    "Logistic Regression": "logit_model.pkl",
}

@st.cache_resource
def load_models():
    models = {}
    for label, filename in MODEL_FILES.items():
        with open(filename, "rb") as f:
            data = pickle.load(f)
        models[label] = data
    return models

models = load_models()
col1, col2 = st.columns([1, 8])

with col1:
    st.image("images/iris_icon.png", width=60)

with col2:
    st.title("Iris Flower Classifier")

st.markdown("Adjust the sliders and each tab will "
            "predict the flower species using a different model.")

# Shared input widgets
@st.cache_data
def get_iris_parameters():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    features = ["sepal length (cm)", "sepal width (cm)",
                "petal length (cm)", "petal width (cm)"]

    params = {}
    for col in features:
        params[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean())
        }

    return params

params = get_iris_parameters()

sepal_length = st.slider(
    "Sepal Length (cm)",
    params["sepal length (cm)"]["min"],
    params["sepal length (cm)"]["max"],
    params["sepal length (cm)"]["mean"]
)

sepal_width = st.slider(
    "Sepal Width (cm)",
    params["sepal width (cm)"]["min"],
    params["sepal width (cm)"]["max"],
    params["sepal width (cm)"]["mean"]
)

petal_length = st.slider(
    "Petal Length (cm)",
    params["petal length (cm)"]["min"],
    params["petal length (cm)"]["max"],
    params["petal length (cm)"]["mean"]
)

petal_width = st.slider(
    "Petal Width (cm)",
    params["petal width (cm)"]["min"],
    params["petal width (cm)"]["max"],
    params["petal width (cm)"]["mean"]
)

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
tabs = st.tabs(list(MODEL_FILES.keys()))

for tab, label in zip(tabs, MODEL_FILES.keys()):
    with tab:
        model = models[label]["model"]
        classes = models[label]["classes"]

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        predicted_class = classes[prediction].capitalize()

        st.markdown(
            f"### Predicted class: <span style='background-color: #b192ff;"
            "color: black; padding: 0px 6px; border-radius: 4px;'>"
            f"{predicted_class}</span>",
            unsafe_allow_html=True
        )

        st.subheader("Confidence")
        for cls, prob in zip(classes, probabilities):
            st.progress(prob, text=f"{cls.capitalize()}: {prob:.1%}")

        predic_type = classes[prediction]

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if predic_type == "setosa":
                st.image("images/iris_setosa.png", caption="Iris Setosa", use_container_width=True)
            elif predic_type == "versicolor":
                st.image("images/iris_versicolor.png", caption="Iris Versicolor", use_container_width=True)
            elif predic_type == "virginica":
                st.image("images/iris_virginica.png", caption="Iris Virginica", use_container_width=True)