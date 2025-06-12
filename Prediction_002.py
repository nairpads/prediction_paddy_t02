import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("🚢 Titanic Survival Prediction + EDA")

# File uploads
train_file = st.file_uploader("Upload train.csv", type=["csv"])
test_file = st.file_uploader("Upload test.csv", type=["csv"])

if train_file and test_file:
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # Normalize column names
    train.columns = train.columns.str.strip().str.lower()
    test.columns = test.columns.str.strip().str.lower()

    # Save PassengerId before dropping
    test_passenger_id = test['passengerid'].copy()

    # Combine for shared preprocessing
    combine = [train, test]
    for dataset in combine:
        if 'age' in dataset.columns:
            dataset['age'].fillna(dataset['age'].median(), inplace=True)
        if 'fare' in dataset.columns:
            dataset['fare'].fillna(dataset['fare'].median(), inplace=True)
        if 'embarked' in dataset.columns:
            dataset['embarked'].fillna(dataset['embarked'].mode()[0], inplace=True)
            dataset['embarked'] = dataset['embarked'].astype(str).str.upper().apply(
                lambda x: {'S': 0, 'C': 1, 'Q': 2}.get(x, -1)
            ).astype(int)
        if 'sex' in dataset.columns:
            dataset['sex'] = dataset['sex'].astype(str).str.lower().map({'male': 0, 'female': 1})
            dataset['sex'].fillna(-1, inplace=True)
            dataset['sex'] = dataset['sex'].astype(int)

    # Drop unused columns
    drop_cols = ['name', 'ticket', 'cabin', 'passengerid']
    train = train.drop(columns=[col for col in drop_cols if col in train.columns])
    test = test.drop(columns=[col for col in drop_cols if col in test.columns])

    # EDA
    st.subheader("🔍 Dataset Preview")
    st.dataframe(train.head())

    st.subheader("📊 Survival Count")
    sns.countplot(x='survived', data=train)
    st.pyplot()

    st.subheader("👩‍🦰🧔 Survival by Gender")
    sns.countplot(x='sex', hue='survived', data=train)
    st.pyplot()

    st.subheader("🛌 Survival by Passenger Class")
    sns.countplot(x='pclass', hue='survived', data=train)
    st.pyplot()

    st.subheader("🎂 Age Distribution with Survival")
    sns.histplot(data=train, x='age', hue='survived', bins=30, kde=True)
    st.pyplot()

    # Model training
    X = train.drop("survived", axis=1)
    y = train["survived"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train, y_train)
    acc_log = accuracy_score(y_val, logreg.predict(X_val))

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_val, rf.predict(X_val))

    # Predictions
    final_predictions = rf.predict(test)
    submission = pd.DataFrame({
        "PassengerId": test_passenger_id,
        "Survived": final_predictions
    })

    # Report
    report = f"""Model Performance Report - Titanic
----------------------------------
Logistic Regression Accuracy: {acc_log:.4f}
Random Forest Accuracy:       {acc_rf:.4f}
Final model used:             Random Forest (n_estimators=100)
"""

    # Feature importance
    st.subheader("🌲 Feature Importances (Random Forest)")
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    sns.barplot(x=feature_importance, y=feature_importance.index)
    st.pyplot()

    # ZIP output
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("submission.csv", submission.to_csv(index=False))
        zip_file.writestr("report.txt", report)

    # Download section
    st.subheader("📦 Download All Results")
    st.download_button(
        label="Download ZIP (submission + report)",
        data=zip_buffer.getvalue(),
        file_name="titanic_results.zip",
        mime="application/zip"
    )

    st.subheader("📄 Report Summary")
    st.code(report)

else:
    st.info("👋 Please upload both training and test CSV files to get started.")
