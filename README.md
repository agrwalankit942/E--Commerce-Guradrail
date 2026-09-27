🛡️ E-Commerce Guardrail

An Explainable Multi-Signal Framework for Fake Review Detection and Suspicious Return Behaviour Prediction

Project type: Artificial Intelligence / Machine Learning / NLP /
E-Commerce Analytics
Interface: Streamlit
Core ideas: Advanced NLP, probability calibration, SHAP
explainability, uncertainty-aware detection, evidence conflict, and
dynamic CRI fusion.

📌 Project in Simple Words

E-commerce platforms face two different risks:

Fake reviews can mislead customers.

Unusual return/cancellation behaviour can create operational and
financial risk.

This project builds two independent detection engines and then combines
their risk outputs into one Guardrail decision.

Easy example

Imagine a customer writes a very normal-looking review, but the same
customer's transaction behaviour shows an unusually high return/refund
pattern.

Instead of saying:

"The review is genuine, so everything is safe."

the Guardrail looks at both signals:

Review Evidence
      ↓
S_rev

Return Behaviour Evidence
      ↓
S_ret

S_rev + S_ret
      ↓
Evidence Conflict / Uncertainty
      ↓
Dynamic CRI
      ↓
Low Risk / Human Review / High Risk

The important point is that the two original datasets are not falsely
joined row-by-row. The framework combines their model-level risk
outputs.

🎯 Project Objectives

Detect suspicious/fake reviews using NLP.

Analyse suspicious customer return behaviour.

Produce interpretable risk scores instead of only class labels.

Calibrate model probabilities where supported.

Detect disagreement between independent risk engines.

Escalate ambiguous cases to human review.

Explain important behavioural factors using SHAP/model feature
importance.

Provide an interactive Streamlit dashboard.

🧠 System Architecture

                    E-COMMERCE GUARDRIAL
                           │
             ┌─────────────┴─────────────┐
             │                           │
       ENGINE 1                     ENGINE 2
   Fake Review NLP             Return Behaviour
             │                           │
      Review Text                   Customer Features
             │                           │
      NLP / Transformer            GBDT / Tree Model
             │                           │
          S_rev                       S_ret
             │                           │
             └─────────────┬─────────────┘
                           │
                  Probability Calibration
                           │
                  Evidence Conflict Check
                           │
                    Uncertainty Analysis
                           │
                    Dynamic CRI Engine
                           │
                 ┌─────────┴─────────┐
                 │                   │
              Lower Risk        Human Review
                 │                   │
                 └───────┬───────────┘
                         │
                   Explainability
                         │
                    Streamlit UI

🔹 Module 1 --- Fake Review Detection

The project notebook uses the master_ecommerce_guardrail.csv dataset.

Dataset

Rows: 100,000

Columns: 58

Missing values: 0

Duplicate rows: 0

Target: is_fake_review

Genuine reviews: 94,100

Fake reviews: 5,900

Fake-review proportion: 5.9%

The notebook reports these dataset properties during its audit.

Signals available in the dataset

Examples include:

Star rating

Verified purchase

Review length

Helpful votes

Reviewer activity

Early-review indicator

Sentiment

Subjectivity

Polarity

Emotion

Product/category information

Seller information

Shipping information

Return-rate information

NLP pipeline

The notebook contains:

Review Text
   ↓
Text preprocessing / synthesized text representation
   ↓
TF-IDF baseline
   ↓
Logistic Regression / Naive Bayes
   ↓
Transformer setup with DistilBERT

The notebook's baseline results include:

Model                                    Accuracy       F1   ROC-AUC

TF-IDF + Logistic Regression               0.9410   0.0000    0.7810
TF-IDF + Naive Bayes                       0.9410   0.0000    0.7793
TF-IDF + SMOTE + Logistic Regression       0.6897   0.2363    0.7803
TF-IDF + SMOTE + Naive Bayes               0.6140   0.2105    0.7792

Important: the high baseline accuracy is affected by class
imbalance; the F1 score shows why accuracy alone should not be used to
judge the fake-review model.

🔹 Module 2 --- Suspicious Return Behaviour

The second module uses transaction-level information to derive customer
behaviour.

The six behavioural features used by the notebook are:

total_orders
total_returns
total_spend
total_refunded
return_ratio
refund_ratio

The notebook trains a Gradient Boosting model for suspicious return
behaviour.

Reported test performance:

Metric       Result

Accuracy     0.9596
F1-Score     0.8273
ROC-AUC      0.9833

Important methodological point

The underlying Online Retail II data does not provide confirmed fraud
labels.

Therefore this project should describe the second output as:

Suspicious Return Behaviour Risk

rather than claiming that every cancellation/return is confirmed fraud.

🔀 Cross-Module Risk Fusion

The two modules remain independent.

Their outputs are combined at the framework level:

[ CRI = \frac{w_{rev}S_{rev}+w_{ret}S_{ret}}{=tex}
{w_{rev}+w_{ret}} ]

Where:

(S_{rev}) = calibrated fake-review risk

(S_{ret}) = calibrated suspicious-return risk

(w_{rev}) = review-risk weight

(w_{ret}) = return-risk weight

The Streamlit sidebar allows these weights to be changed dynamically.

🎯 Uncertainty-Aware Detection

A normal classifier forces:

Fake
or
Genuine

This project adds a third decision path:

Low/clear risk
      ↓
Normal action

Ambiguous / conflicting evidence
      ↓
HUMAN REVIEW

This is important because a model can be uncertain even when it produces
a numerical probability.

The dashboard therefore considers:

probability uncertainty

distance from the CRI decision threshold

cross-module evidence conflict

⚠️ Evidence Conflict Detection

The Guardrail checks whether the two engines disagree strongly.

Example:

Review risk       = Low
Return risk       = High
             ↓
      Evidence Conflict
             ↓
       Human Review

This prevents the system from relying on only one signal.

🔎 Explainability

For the return-behaviour model, the notebook uses SHAP/TreeSHAP to
identify feature contributions.

Example features:

return_ratio
refund_ratio
total_refunded
total_returns
total_orders
total_spend

The Streamlit dashboard visualizes these contributions/feature
importance so the user can understand why a risk score was produced.

📊 Notebook Results

Confusion Matrices

The following figure is taken from the project's notebook evaluation
section.



The notebook reports separate evaluation for:

Fake Review Detection

Suspicious Return Behaviour

The fake-review evaluation uses a 0.60 threshold in that analysis, while
the return module is also evaluated using a 0.60 threshold.

Return Behaviour Feature Importance



This graph shows the relative contribution/importance of the six
behavioural variables used by the return-risk model.

🖥️ Streamlit Dashboard

The dashboard is designed around the project workflow.

1. Overall Project

Explains:

project objective

two independent engines

Guardrail architecture

CRI concept

uncertainty

evidence conflict

explainability

2. Fake Review Analyzer

User enters review text and receives:

review-risk score

uncertainty

model confidence

risk interpretation

3. Return Behaviour Analyzer

User enters:

total orders

total returns

total spend

total refunded

The dashboard automatically calculates:

return ratio

refund ratio

suspicious-return score

uncertainty

feature importance

4. Guardrail Fusion

This is the main decision-support screen.

It shows:

S_rev
S_ret
CRI
Uncertainty
Evidence Conflict
Final Risk Tier
Human Review Decision

5. Analytics

Supports interactive:

target-distribution pie charts

feature distributions

categorical charts

model-derived charts

6. Explainability

Shows:

TF-IDF model coefficients

return-model feature importance

SHAP availability/status

7. Model & Data Status

Displays:

available artifacts

model types

feature count

Transformer status

reproducibility information

🚀 How to Run

Clone/download the project and install dependencies:

pip install -r requirements.txt

Then start Streamlit:

streamlit run app.py

📦 Main Project Structure

E-Commerce-Guardrail/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── master_ecommerce_guardrail.csv
│   └── online_retail_II.xlsx
│
├── models/
│   ├── fake_review_model.pkl
│   └── return_model.pkl
│
├── notebooks/
│   └── Guradrail.ipynb
│
└── figures/
    ├── confusion_matrices.png
    └── return_feature_importance.png

⚠️ Reproducibility & Research Notes

1. DistilBERT

The notebook contains DistilBERT architecture setup and later loads:

saved_distilbert_model
saved_distilbert_tokenizer

However, the visible training loop in the notebook is commented out.
Therefore, a final research paper should not claim a verified
fine-tuned DistilBERT result unless a completed training run with
recorded loss and evaluation metrics is performed.

2. Calibration

The Streamlit implementation includes probability-calibration logic
using:

Sigmoid/Platt scaling

Isotonic calibration

Calibration should be evaluated using metrics such as the Brier score
and reliability curves.

3. Return labels

The return module represents suspicious behaviour rather than confirmed
fraud because the source data does not contain a ground-truth fraud
label.

4. Synthetic/illustrative dashboard values

Any dashboard-generated training demonstrations, simulated values, or
hard-coded illustrative summaries should not be reported as measured
experimental results in the research paper.

🔬 Research Contribution Direction

The project is designed around four main research-oriented ideas:

Uncertainty-aware risk detection

Explainable multi-signal risk assessment

Cross-module evidence conflict

Risk fusion through a dynamic CRI

The final research novelty should be established after the literature
review rather than assuming that the combination is automatically
unprecedented.

📚 Current Project Status

Dataset Audit                         ✅
Fake Review Module                    ✅
Return Behaviour Module               ✅
Risk Fusion / CRI                     ✅
Evidence Conflict                     ✅
Uncertainty Handling                  ✅
Explainability                        ✅
Streamlit Dashboard                   ✅
Research Figures                      ✅
Advanced DistilBERT Training          ⚠️ Requires verified training run
Final Calibration Validation          ⚠️ Needs final experimental validation

👨‍💻 Project Summary

E-Commerce Guardrail is an explainable decision-support framework
for e-commerce risk analysis.

Instead of depending on a single prediction, it evaluates review
evidence and customer return behaviour independently, converts
them into risk signals, checks for uncertainty and disagreement, and
combines the evidence into a dynamic Combined Risk Index (CRI).

The goal is not to automatically accuse a customer or reviewer of fraud.
The goal is to prioritize suspicious or ambiguous cases for
appropriate human review while providing interpretable evidence for
the decision.
