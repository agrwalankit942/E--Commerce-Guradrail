# 🛡️ E-Commerce Guardrail: Multi-Modal Risk Fusion & Fraud Prevention Engine

An advanced AI/ML Guardrail System designed for e-commerce platforms. It combines **Natural Language Processing (NLP)** and **Gradient Boosted Decision Trees (GBDT)** to simultaneously catch **Fake/Deceptive Reviews** and **Suspicious Return Behavior**, backed by **Probability Calibration** and **Explainable AI (TreeSHAP)**.

---

## 🌟 Key Highlights & Novelties

1. **Dual-Engine Risk Fusion (CRI):** Calculates a unified **Customer Risk Index (CRI)** combining text-based deception scores ($S_{rev}$) and behavioral transaction scores ($S_{ret}$).
2. **Probability Calibration:** Integrates **Platt Scaling (Sigmoid)** and **Isotonic Regression** to eliminate overconfident raw ML predictions and improve probabilistic accuracy (lowering Brier Score).
3. **Explainable AI (TreeSHAP):** Provides prediction-level transparency with SHAP feature attribution charts showing *why* a customer or review was flagged.
4. **Human Escalation Queue:** Automatically detects conflicts between text reviews and return behaviors ($\vert{}S_{rev} - S_{ret}\vert{} \ge 0.45$) or high uncertainty zones, routing cases to human compliance teams.
5. **Dynamic Action Matrix:** Automatically executes business logic (Auto-Approve, Soft Warning, Shadowban, COD Restriction) based on risk severity.

---

## 🏗️ System Architecture

```text
                                [ User Transaction / Review Input ]
                                                 │
          ┌──────────────────────────────────────┴──────────────────────────────────────┐
          ▼                                                                             ▼
[ Engine 1: DistilBERT / NLP ]                                       [ Engine 2: XGBoost / GBDT Return Model ]
  • Feature Input: Review Text, Ratings, Sentiment                     • Feature Input: Order Frequency, Return Ratio, Spend
  • Calibration: Platt Scaling (Sigmoid)                               • Calibration: Isotonic Regression
  • Output: Calibrated Score (S_rev)                                   • Output: Calibrated Score (S_ret)
          │                                                                             │
          └──────────────────────────────────────┬──────────────────────────────────────┘
                                                 ▼
                             [ Dynamic Risk Fusion & Decision Engine ]
                             Formula: CRI = (w_rev * S_rev + w_ret * S_ret) / (w_rev + w_ret)
                                                 │
         ┌───────────────────────────────────────┼───────────────────────────────────────┐
         ▼                                       ▼                                       ▼
[ Low Risk (< 0.35) ]                  [ Medium / Uncertainty ]                [ Critical Risk (≥ 0.65) ]
• Auto-Approve Review                  • Soft Warning                          • Shadowban Review
• Instant Refund Approval              • Escalated to Human Review             • Restrict COD / Hold Payout
