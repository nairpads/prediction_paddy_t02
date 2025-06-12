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

#st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("🚢 Titanic Survival Prediction + EDA")

# Upload section
train_file = st.file_uploader("Upload train.csv", type=["csv"])
test_file = st.file_uploader("Upload test.csv", type=["csv"])

if train_file and test_file:
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # Combine datasets for preprocessing
    combine = [train, test]
    for dataset in combine:
        dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
        dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    test['Fare'].fillna(test['Fare'].median(), inplace=True)

    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1}).astype(int)
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    test_passenger_id = test['PassengerId']
    train = train.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    test = test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    # Show data overview
    st.subheader("🔍 Dataset Preview")
    st.dataframe(train.head())

    # EDA: Survival Count
    st.subheader("📊 Survival Count")
    sns.countplot(x='Survived', data=train)
    st.pyplot()

    # EDA: Survival by Gender
    st.subheader("👩‍🦰🧔 Survival by Gender")
    sns.countplot(x='Sex', hue='Survived', data=train)
    st.pyplot()

    # EDA: Survival by Class
    st.subheader("🛌 Survival by Passenger Class")
    sns.countplot(x='Pclass', hue='Survived', data=train)
    st.pyplot()

    # EDA: Age Distribution
    st.subheader("🎂 Age Distribution with Survival")
    sns.histplot(data=train, x='Age', hue='Survived', bins=30, kde=True)
    st.pyplot()

    # Split features/target
    X = train.drop("Survived", axis=1)
    y = train["Survived"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models
    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train, y_train)
    y_pred_log = logreg.predict(X_val)
    acc_log = accuracy_score(y_val, y_pred_log)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_val)
    acc_rf = accuracy_score(y_val, y_pred_rf)

    # Final prediction
    final_predictions = rf.predict(test)
    submission = pd.DataFrame({
        "PassengerId": test_passenger_id,
        "Survived": final_predictions
    })

    # Create model report
    report = f"""
    Model Performance Report - Titanic
    ----------------------------------
    Logistic Regression Accuracy: {acc_log:.4f}
    Random Forest Accuracy:       {acc_rf:.4f}
    Final model used:             Random Forest (n_estimators=100)
    """

    # Feature importance plot
    st.subheader("🌲 Feature Importances (Random Forest)")
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    sns.barplot(x=feature_importance, y=feature_importance.index)
    st.pyplot()

    # ZIP output
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("submission.csv", submission.to_csv(index=False))
        zip_file.writestr("report.txt", report)

    # Download ZIP
    st.subheader("📦 Download All Results")
    st.download_button(
        label="Download ZIP (submission + report)",
        data=zip_buffer.getvalue(),
        file_name="titanic_results.zip",
        mime="application/zip"
    )

    # Show report
    st.subheader("📄 Report Summary")
    st.code(report)
