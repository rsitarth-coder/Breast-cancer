import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="ML Classification Comparison", layout="wide")
st.title("Classification Model-Breast Cancer")
st.caption("Logistic Regression, Decision Tree, KNN, Naive Bayes and SVM, evaluated on one dataset.")


@st.cache_data
def load_builtin():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df["target"] = df["target"].map({0: "malignant", 1: "benign"})
    return df, "target"


def build_models(n_classes):
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "SVM (RBF)": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", probability=(n_classes > 2), random_state=42)
        ),
    }


def get_scores(model, X, n_classes):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if n_classes == 2 else p
    return model.decision_function(X)


@st.cache_data(show_spinner="Training models...")
def run_all(X, y, test_size, seed):
    n_classes = len(np.unique(y))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    avg = "binary" if n_classes == 2 else "weighted"
    rows, outputs = [], {}
    for name, model in build_models(n_classes).items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        score = get_scores(model, X_te, n_classes)
        auc = (roc_auc_score(y_te, score) if n_classes == 2
               else roc_auc_score(y_te, score, multi_class="ovr"))
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "AUC": auc,
            "Precision": precision_score(y_te, pred, average=avg, zero_division=0),
            "Recall": recall_score(y_te, pred, average=avg, zero_division=0),
            "F1": f1_score(y_te, pred, average=avg, zero_division=0),
            "MCC": matthews_corrcoef(y_te, pred),
        })
        outputs[name] = {"pred": pred, "score": score}
    results = pd.DataFrame(rows).set_index("Model")
    return results, outputs, y_te, n_classes


st.sidebar.header("Settings")
source = st.sidebar.radio("Dataset", ["Built-in: Breast Cancer (UCI)", "Upload CSV"])

if source == "Upload CSV":
    file = st.sidebar.file_uploader("CSV file", type="csv")
    if file is None:
        st.info("Upload a CSV with at least 12 features and 500 rows, then pick the target column.")
        st.stop()
    df = pd.read_csv(file)
    target = st.sidebar.selectbox("Target column", df.columns, index=len(df.columns) - 1)
else:
    df, target = load_builtin()

test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

df = df.dropna()
X = pd.get_dummies(df.drop(columns=[target]), drop_first=True).astype(float)
label_enc = LabelEncoder()
y = pd.Series(label_enc.fit_transform(df[target]), index=df.index)
class_names = [str(c) for c in label_enc.classes_]

if X.shape[0] < 500 or X.shape[1] < 12:
    st.warning(f"Dataset has {X.shape[0]} rows and {X.shape[1]} features; "
               "the assignment needs at least 500 rows and 12 features.")

results, outputs, y_te, n_classes = run_all(X, y, test_size, int(seed))

tab1, tab2, tab3 = st.tabs(["Metrics comparison", "Model details", "Dataset"])

with tab1:
    st.subheader("Evaluation metrics (test set)")
    st.dataframe(results.style.format("{:.4f}").highlight_max(axis=0, color="#2e7d32"))
    st.caption("Highlighted cell = best model for that metric. "
               + ("" if n_classes == 2 else "Multi-class: weighted precision/recall/F1, one-vs-rest AUC."))
    st.bar_chart(results.drop(columns=["MCC"]).T)
    best = results["MCC"].idxmax()
    st.success(f"Best model by MCC: **{best}** ({results.loc[best, 'MCC']:.4f})")

with tab2:
    model_name = st.selectbox("Select model", list(results.index))
    out = outputs[model_name]
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_te, out["pred"])
        cm_df = pd.DataFrame(
            [(class_names[i], class_names[j], int(cm[i, j]))
             for i in range(len(class_names)) for j in range(len(class_names))],
            columns=["Actual", "Predicted", "Count"],
        )
        base = alt.Chart(cm_df).encode(
            x=alt.X("Predicted:N", sort=class_names, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Actual:N", sort=class_names),
        )
        heat = base.mark_rect().encode(
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None)
        )
        text = base.mark_text(fontSize=16).encode(
            text="Count:Q",
            color=alt.condition(alt.datum.Count > cm.max() / 2,
                                alt.value("white"), alt.value("black")),
        )
        st.altair_chart((heat + text).properties(height=300), use_container_width=True)

    with c2:
        if n_classes == 2:
            st.markdown("**ROC curve**")
            fpr, tpr, _ = roc_curve(y_te, out["score"])
            auc_val = roc_auc_score(y_te, out["score"])
            roc_df = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})
            diag = pd.DataFrame({"False Positive Rate": [0, 1], "True Positive Rate": [0, 1]})
            roc_line = alt.Chart(roc_df).mark_line(strokeWidth=2).encode(
                x=alt.X("False Positive Rate:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("True Positive Rate:Q", scale=alt.Scale(domain=[0, 1])),
            )
            diag_line = alt.Chart(diag).mark_line(strokeDash=[4, 4], color="gray").encode(
                x="False Positive Rate:Q", y="True Positive Rate:Q"
            )
            st.altair_chart((roc_line + diag_line).properties(height=300),
                            use_container_width=True)
            st.caption(f"{model_name}: AUC = {auc_val:.4f}")
        else:
            st.info("ROC curve is shown for binary problems only.")

    st.markdown("**Classification report**")
    report = classification_report(y_te, out["pred"], target_names=class_names,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4))

with tab3:
    st.write(f"**Rows:** {X.shape[0]}  |  **Features:** {X.shape[1]}  |  **Classes:** {n_classes}")
    st.write("**Class distribution**")
    st.bar_chart(df[target].value_counts())
    st.dataframe(df.head(50))
