from sklearn.linear_model import LogisticRegression
from src.model_utils import log_metrics_and_params

def test_log_metrics_and_params_runs_without_error():
    model = LogisticRegression(max_iter=1000)
    X = [[0, 1], [1, 0], [1, 1], [0, 0]]
    y = [0, 1, 1, 0]
    model.fit(X, y)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    try:
        log_metrics_and_params(model, y, y_pred, y_prob)
    except Exception as e:
        assert False, f"log_metrics_and_params failed: {e}"
