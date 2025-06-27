📘 Credit Scoring Business Understanding
1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
  -Basel II requires banks to:

  -Measure credit risk accurately

  -Justify internal ratings systems

The Basel II Accord emphasizes accurate, transparent risk measurement. This requires credit scoring models to be interpretable, well-documented, and auditable so that both internal teams and regulators can understand and validate how credit decisions are made.

This places a strong emphasis on using interpretable models and maintaining clear documentation. While modern ML models offer better accuracy, the Basel II framework nudges institutions toward responsible AI—ensuring models are not black boxes, but rather tools that support informed, compliant credit decisions.

2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In the absence of a true "default" label (e.g., "customer missed 90+ days of payments"), we often build a proxy variable using assumptions, like:

  -Late payment past a threshold (e.g., >60 days)

  -Account charged-off

  -Collections activity

Why We Use a Proxy Variable
Due to the lack of a direct "default" label, we must define a proxy (e.g., based on fraud status, repayment patterns, or RFM behavior) to train our model. While necessary, this introduces risks like label noise, bias, and misclassification, which can lead to financial or regulatory issues if not handled carefully.

Creating a proxy is necessary to:

 -Enable supervised learning using available historical data.

 -Simulate credit risk likelihood for model training and evaluation.

 -Guide decisions on loan approvals, limits, and durations

3. Model Trade-offs: Simplicity vs. Performance

Simple models (like Logistic Regression with WoE) are transparent, easy to deploy, and regulatory-friendly.

Complex models (like Gradient Boosting) often perform better but are harder to interpret and justify.
In a financial setting, interpretability and compliance often outweigh marginal gains in accuracy.
