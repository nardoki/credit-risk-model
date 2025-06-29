import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv("data/processed/with_labels.csv")

# Drop identifiers and set target
X = data.drop(columns=["is_high_risk", "CustomerId"])
y = data["is_high_risk"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and parameters
models = {
    "logistic_regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            'model__C': [0.01, 0.1, 1, 10]
        }
    },
    "random_forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            'model__n_estimators': [50, 100],
            'model__max_depth': [3, 5, 10]
        }
    }
}

mlflow.set_experiment("credit-risk-model")

for model_name, config in models.items():
    with mlflow.start_run(run_name=model_name):
        # Create pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', config['model'])
        ])

        clf = GridSearchCV(pipe, config['params'], cv=3, scoring='roc_auc')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        # Log metrics
        mlflow.log_params(clf.best_params_)
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        })

        mlflow.sklearn.log_model(clf.best_estimator_, "model", registered_model_name=model_name)

        print(f"Finished training {model_name}")
