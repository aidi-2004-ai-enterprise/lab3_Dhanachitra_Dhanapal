# train.py

"""
Train an XGBoost classifier on the Seaborn penguins dataset
and save the model and label encoder classes.
"""

import os
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix


def load_and_preprocess_data() -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """Load, clean, encode, and split the penguins dataset."""
    df = sns.load_dataset("penguins")
    print("Initial dataset shape:", df.shape)

    # Drop rows with missing values
    df = df.dropna(subset=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex", "island"])

    # Encode target variable
    le = LabelEncoder()
    df["species"] = le.fit_transform(df["species"])

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=["sex", "island"])

    # Fix column casing to match Enum values in FastAPI
    df.columns = df.columns.str.replace("sex_female", "sex_Female")
    df.columns = df.columns.str.replace("sex_male", "sex_Male")

    X = df.drop("species", axis=1)
    y = df["species"]

    return X, y, le


def train_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """Train an XGBoost classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        use_label_encoder=False,
        eval_metric="mlogloss",
        verbosity=0
    )
    model.fit(X_train, y_train)

    print("\nTrain classification report:")
    print(classification_report(y_train, model.predict(X_train)))
    print("\nTest classification report:")
    print(classification_report(y_test, model.predict(X_test)))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, model.predict(X_test)))
    print("Train F1 Score:", f1_score(y_train, model.predict(X_train), average='weighted'))
    print("Test F1 Score:", f1_score(y_test, model.predict(X_test), average='weighted'))

    return model


def save_model_and_encoder(model: xgb.XGBClassifier, label_encoder: LabelEncoder) -> None:
    """Save the trained model and label encoder class names to disk."""
    os.makedirs("app/data", exist_ok=True)
    model.save_model("app/data/model.json")
    pd.DataFrame({"species": label_encoder.classes_}).to_json("app/data/label_encoder_classes.json", orient="records")
    print("Model and label encoder saved to app/data/")


if __name__ == "__main__":
    X, y, le = load_and_preprocess_data()
    model = train_model(X, y)
    save_model_and_encoder(model, le)
