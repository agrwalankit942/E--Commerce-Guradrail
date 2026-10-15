
# ============================================================
# E-COMMERCE GUARDRAIL - COMPLETE STREAMLIT DASHBOARD
# Fake Review Detection + Suspicious Return Risk Prediction
# TF-IDF + Logistic Regression | Gradient Boosting
# ============================================================

from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="E-Commerce Guardrail",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "master_ecommerce_guardrail.csv"
FAKE_MODEL_FILE = BASE_DIR / "fake_review_model.pkl"
RETURN_MODEL_FILE = BASE_DIR / "return_model.pkl"

st.title("🛡️ E-Commerce Guardrail")
st.caption(
    "Fake Review Detection | Return Risk Prediction | "
    "Risk Assessment Dashboard"
)


# ============================================================
# 2. LOAD DATA AND SAVED MODELS
# ============================================================

@st.cache_data
def load_dataset(file_path):
    return pd.read_csv(file_path)


@st.cache_resource
def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE.name}")
    st.stop()

if not FAKE_MODEL_FILE.exists():
    st.error(
        f"Fake review model not found: {FAKE_MODEL_FILE.name}"
    )
    st.stop()

if not RETURN_MODEL_FILE.exists():
    st.error(
        f"Return model not found: {RETURN_MODEL_FILE.name}"
    )
    st.stop()

try:
    df = load_dataset(str(DATA_FILE))

    fake_bundle = load_pickle(str(FAKE_MODEL_FILE))
    return_bundle = load_pickle(str(RETURN_MODEL_FILE))

    # Expected fake review bundle:
    # {"model": model_lr, "tfidf": tfidf}
    if not isinstance(fake_bundle, dict):
        raise ValueError(
            "fake_review_model.pkl must contain a dictionary "
            "with 'model' and 'tfidf' keys."
        )

    fake_model = fake_bundle["model"]
    tfidf = fake_bundle["tfidf"]

    # Support either a dictionary bundle or a directly saved model.
    if isinstance(return_bundle, dict):
        return_model = return_bundle.get(
            "model", return_bundle
        )

        return_features = return_bundle.get(
            "features",
            [
                "total_orders",
                "total_returns",
                "total_spend",
                "total_refunded",
                "return_ratio",
                "refund_ratio",
            ],
        )
    else:
        return_model = return_bundle

        return_features = [
            "total_orders",
            "total_returns",
            "total_spend",
            "total_refunded",
            "return_ratio",
            "refund_ratio",
        ]

except Exception as exc:
    st.error(
        f"Could not load the dataset or models: {exc}"
    )
    st.stop()


# ============================================================
# 3. VALIDATE DATASET COLUMNS
# ============================================================

required_columns = [
    "star_rating",
    "sentiment",
    "emotion",
    "quality_aspect",
    "is_fake_review",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(
        f"Missing required dataset columns: {missing_columns}"
    )
    st.stop()


# ============================================================
# 4. PREDICTION FUNCTIONS
# ============================================================

def structured_review_text(
    rating,
    sentiment,
    emotion,
    quality,
):
    """
    This format must match the representation used when
    training the saved TF-IDF review classifier.
    """

    return (
        f"Rating: {rating} stars. "
        f"Sentiment: {sentiment}. "
        f"Emotion: {emotion}. "
        f"Aspect Quality: {quality}"
    )


def predict_fake_review(
    review_text,
    rating,
    sentiment,
    emotion,
    quality,
):
    """
    Predict using the saved TF-IDF + Logistic Regression model.

    Note: The current model uses structured review attributes.
    The free-text review_text input is not used by this model.
    """

    text = structured_review_text(
        rating,
        sentiment,
        emotion,
        quality,
    )

    vector = tfidf.transform([text])
    probabilities = fake_model.predict_proba(vector)[0]

    classes = list(fake_model.classes_)

    if 1 not in classes:
        raise ValueError(
            "The fake-review model does not contain class 1."
        )

    fake_index = classes.index(1)
    fake_probability = float(probabilities[fake_index])

    return fake_probability, "TF-IDF + Logistic Regression"


def predict_return_risk(
    orders,
    returns,
    spend,
    refunded,
):
    """
    Build the customer behavior features expected by the
    saved Gradient Boosting return-risk model.
    """

    feature_values = {
        "total_orders": orders,
        "total_returns": returns,
        "total_spend": spend,
        "total_refunded": refunded,
        "return_ratio": returns / (orders + 1e-5),
        "refund_ratio": refunded / (spend + 1e-5),
    }

    missing = [
        col for col in return_features
        if col not in feature_values
    ]

    if missing:
        raise ValueError(
            f"Unsupported return-model features: {missing}"
        )

    X = pd.DataFrame(
        [[feature_values[col] for col in return_features]],
        columns=return_features,
    )

    probabilities = return_model.predict_proba(X)[0]
    classes = list(return_model.classes_)

    if 1 not in classes:
        raise ValueError(
            "The return model does not contain risk class 1."
        )

    risk_probability = float(
        probabilities[classes.index(1)]
    )

    return risk_probability, feature_values


def calculate_combined_risk(review_p, return_p):
    """
    Calculate the equal-weight combined risk index.
    This is not a calibrated probability of fraud.
    """

    combined_score = 0.5 * review_p + 0.5 * return_p

    disagreement = (
        (review_p < 0.25 and return_p > 0.70)
        or (review_p > 0.75 and return_p < 0.20)
    )

    if combined_score >= 0.70:
        category = "HIGH RISK"
        recommendation = (
            "Prioritize this case for investigation and verify "
            "supporting evidence before taking action."
        )

    elif combined_score >= 0.40 or disagreement:
        category = "MEDIUM / UNCERTAIN"
        recommendation = (
            "Send this case for human review. The available "
            "signals do not justify an automatic decision."
        )

    else:
        category = "LOW RISK"
        recommendation = (
            "No strong combined risk signal was detected. "
            "Continue normal checks."
        )

    return (
        combined_score,
        category,
        recommendation,
        disagreement,
    )


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🛡️ Navigation")

page = st.sidebar.radio(
    "Select Dashboard Page",
    [
        "Executive Overview",
        "Analyze a Case",
        "Model Insights",
        "Dataset Explorer",
    ],
)

st.sidebar.divider()

st.sidebar.write(
    "**Fake Review Model:** TF-IDF + Logistic Regression"
)

st.sidebar.write(
    "**Return Risk Model:** Gradient Boosting"
)

st.sidebar.caption(
    "Research prototype: predictions are risk estimates, "
    "not proof of fraudulent activity."
)


# ============================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.header("📊 Executive Overview")

    st.write(
        "Explore review authenticity labels, sentiment, and "
        "rating distributions in the dataset."
    )

    total_reviews = len(df)

    fake_labels = pd.to_numeric(
        df["is_fake_review"],
        errors="coerce",
    )

    fake_rate = (
        fake_labels.mean() * 100
        if fake_labels.notna().any()
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Review Records",
        f"{total_reviews:,}",
    )

    col2.metric(
        "Fake-Label Rate",
        f"{fake_rate:.2f}%",
    )

    if "product_id" in df.columns:
        col3.metric(
            "Unique Products",
            f"{df['product_id'].nunique():,}",
        )
    else:
        col3.metric("Unique Products", "N/A")

    if "user_id" in df.columns:
        col4.metric(
            "Unique Reviewers",
            f"{df['user_id'].nunique():,}",
        )
    else:
        col4.metric("Unique Reviewers", "N/A")

    st.divider()

    # --------------------------------------------------------
    # PIE CHART 1: REVIEW AUTHENTICITY
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:
        st.subheader("🥧 Review Authenticity Labels")

        label_counts = (
            fake_labels.map({
                0: "Real-labelled",
                1: "Fake-labelled",
            })
            .fillna("Other / Missing")
            .value_counts()
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.pie(
            label_counts.values,
            labels=label_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white"},
        )

        ax.axis("equal")

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)

    # --------------------------------------------------------
    # PIE CHART 2: SENTIMENT
    # --------------------------------------------------------

    with right:
        st.subheader("🥧 Sentiment Distribution")

        sentiment_counts = (
            df["sentiment"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white"},
        )

        ax.axis("equal")

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)

    st.divider()

    # --------------------------------------------------------
    # RATING DISTRIBUTION
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:
        st.subheader("⭐ Rating Distribution")

        rating_counts = (
            df["star_rating"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(rating_counts)

    # --------------------------------------------------------
    # AUTHENTICITY LABELS BY RATING
    # --------------------------------------------------------

    with right:
        st.subheader("Review Labels by Rating")

        cross_tab = pd.crosstab(
            df["star_rating"],
            df["is_fake_review"].map({
                0: "Real",
                1: "Fake",
            }),
        )

        st.bar_chart(cross_tab)

    st.info(
        "These charts describe dataset labels and sentiment. "
        "They do not represent live model predictions or "
        "verified real-world fraud rates."
    )


# ============================================================
# PAGE 2: ANALYZE A CASE
# ============================================================

elif page == "Analyze a Case":

    st.header("🔍 Analyze a Case")

    st.write(
        "Enter the review attributes and the customer's return "
        "history to calculate experimental risk estimates."
    )

    # --------------------------------------------------------
    # REVIEW INFORMATION
    # --------------------------------------------------------

    st.subheader("A. Review Information")

    left, right = st.columns(2)

    sentiment_options = sorted(
        df["sentiment"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    ) or ["Unknown"]

    emotion_options = sorted(
        df["emotion"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    ) or ["Unknown"]

    quality_options = sorted(
        df["quality_aspect"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    ) or ["Unknown"]

    with left:
        review_text = st.text_area(
            "Review Text",
            value="The product quality is excellent.",
            height=120,
        )

        rating = st.selectbox(
            "Star Rating",
            [1, 2, 3, 4, 5],
            index=4,
        )

    with right:
        sentiment = st.selectbox(
            "Sentiment",
            sentiment_options,
        )

        emotion = st.selectbox(
            "Emotion",
            emotion_options,
        )

        quality = st.selectbox(
            "Quality Aspect",
            quality_options,
        )

    st.caption(
        "The saved TF-IDF model uses rating, sentiment, emotion, "
        "and quality attributes. It does not analyze the free-text "
        "review box."
    )

    st.divider()

    # --------------------------------------------------------
    # CUSTOMER RETURN HISTORY
    # --------------------------------------------------------

    st.subheader("B. Customer Return and Refund History")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        orders = st.number_input(
            "Total Orders",
            min_value=0,
            value=10,
            step=1,
        )

    with col2:
        returns = st.number_input(
            "Total Returns",
            min_value=0,
            value=2,
            step=1,
        )

    with col3:
        spend = st.number_input(
            "Total Spend",
            min_value=0.0,
            value=1000.0,
            step=100.0,
        )

    with col4:
        refunded = st.number_input(
            "Total Refunded",
            min_value=0.0,
            value=150.0,
            step=50.0,
        )

    if returns > orders:
        st.warning(
            "Total returns exceed total orders. "
            "Please verify the entered values."
        )

    if refunded > spend:
        st.warning(
            "Total refunded exceeds total spend. "
            "Please verify the entered values."
        )

    # --------------------------------------------------------
    # RUN PREDICTIONS
    # --------------------------------------------------------

    if st.button(
        "🔎 Analyze Risk",
        type="primary",
        use_container_width=True,
    ):

        try:
            review_probability, review_engine = (
                predict_fake_review(
                    review_text,
                    rating,
                    sentiment,
                    emotion,
                    quality,
                )
            )

            return_probability, feature_values = (
                predict_return_risk(
                    orders,
                    returns,
                    spend,
                    refunded,
                )
            )

            (
                combined_score,
                category,
                recommendation,
                disagreement,
            ) = calculate_combined_risk(
                review_probability,
                return_probability,
            )

            st.divider()
            st.subheader("📌 Risk Assessment")

            if category == "HIGH RISK":
                st.error(category)

            elif category == "MEDIUM / UNCERTAIN":
                st.warning(category)

            else:
                st.success(category)

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Fake-Review Score",
                f"{review_probability:.1%}",
            )

            col2.metric(
                "Return-Risk Score",
                f"{return_probability:.1%}",
            )

            col3.metric(
                "Combined Risk Index",
                f"{combined_score:.1%}",
            )

            st.progress(
                float(np.clip(combined_score, 0, 1))
            )

            st.markdown("### Recommended Next Step")
            st.write(recommendation)

            st.caption(f"Review model used: {review_engine}")

            if disagreement:
                st.warning(
                    "The two model scores disagree strongly. "
                    "Human review is especially important."
                )

            st.subheader("Return Model Input Features")

            feature_table = pd.DataFrame({
                "Feature": list(feature_values.keys()),
                "Value": list(feature_values.values()),
            })

            st.dataframe(
                feature_table,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "The combined index is an equal-weight score, "
                "not a calibrated probability of fraud."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ============================================================
# PAGE 3: MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":

    st.header("🧠 Model Insights")

    st.subheader("System Architecture")

    st.markdown("""
    **1. Fake Review Detection**
    - TF-IDF converts structured review attributes into text features.
    - Logistic Regression estimates the fake-review score.

    **2. Suspicious Return Prediction**
    - Gradient Boosting analyzes customer-level return/refund features.

    **3. Risk Fusion**
    - The two scores are combined using equal weights.

    **4. Decision Support**
    - The resulting score is categorized as low, medium/uncertain,
      or high risk.
    """)

    st.divider()

    # --------------------------------------------------------
    # RETURN MODEL FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("📊 Return Model Feature Importance")

    if hasattr(return_model, "feature_importances_"):

        importance_values = return_model.feature_importances_

        if len(importance_values) == len(return_features):

            importance_df = pd.DataFrame({
                "Feature": return_features,
                "Importance": importance_values,
            }).sort_values("Importance")

            fig, ax = plt.subplots(figsize=(9, 4))

            ax.barh(
                importance_df["Feature"],
                importance_df["Importance"],
            )

            ax.set_xlabel("Relative Importance")
            ax.set_title(
                "Gradient Boosting Feature Importance"
            )

            ax.grid(axis="x", alpha=0.25)

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True,
            )

            plt.close(fig)

        else:
            st.warning(
                "The feature names do not match the number of "
                "feature-importance values in the saved model."
            )

    else:
        st.info(
            "The loaded return model does not expose "
            "feature_importances_."
        )

    st.divider()

    # --------------------------------------------------------
    # OPTIONAL TF-IDF MODEL DIAGNOSTIC
    # --------------------------------------------------------

    st.subheader("TF-IDF Classifier Diagnostic")

    st.caption(
        "This optional diagnostic evaluates the loaded TF-IDF "
        "model on the dataset. It is not independent test-set "
        "performance."
    )

    with st.expander("Calculate TF-IDF metrics"):

        if st.button("Calculate TF-IDF Metrics"):

            try:
                from sklearn.metrics import (
                    accuracy_score,
                    precision_score,
                    recall_score,
                    f1_score,
                    confusion_matrix,
                )

                valid = df.dropna(
                    subset=[
                        "star_rating",
                        "sentiment",
                        "emotion",
                        "quality_aspect",
                        "is_fake_review",
                    ]
                ).copy()

                texts = [
                    structured_review_text(
                        rating_value,
                        sentiment_value,
                        emotion_value,
                        quality_value,
                    )
                    for (
                        rating_value,
                        sentiment_value,
                        emotion_value,
                        quality_value,
                    ) in zip(
                        valid["star_rating"],
                        valid["sentiment"],
                        valid["emotion"],
                        valid["quality_aspect"],
                    )
                ]

                y_true = valid["is_fake_review"].astype(int)

                X = tfidf.transform(texts)
                y_pred = fake_model.predict(X)

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Accuracy",
                    f"{accuracy_score(y_true, y_pred):.3f}",
                )

                col2.metric(
                    "Precision",
                    f"{precision_score(y_true, y_pred, zero_division=0):.3f}",
                )

                col3.metric(
                    "Recall",
                    f"{recall_score(y_true, y_pred, zero_division=0):.3f}",
                )

                col4.metric(
                    "F1 Score",
                    f"{f1_score(y_true, y_pred, zero_division=0):.3f}",
                )

                cm = confusion_matrix(
                    y_true,
                    y_pred,
                    labels=[0, 1],
                )

                fig, ax = plt.subplots(figsize=(5, 4))

                ax.imshow(cm)

                ax.set_xticks(
                    [0, 1],
                    labels=["Real", "Fake"],
                )

                ax.set_yticks(
                    [0, 1],
                    labels=["Real", "Fake"],
                )

                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("Actual Label")
                ax.set_title("TF-IDF Confusion Matrix")

                for i in range(2):
                    for j in range(2):
                        ax.text(
                            j,
                            i,
                            str(cm[i, j]),
                            ha="center",
                            va="center",
                        )

                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

                st.warning(
                    "These metrics are measured on the loaded dataset. "
                    "Use a held-out test set for independent evaluation."
                )

            except Exception as exc:
                st.error(
                    f"Metric calculation failed: {exc}"
                )


# ============================================================
# PAGE 4: DATASET EXPLORER
# ============================================================

elif page == "Dataset Explorer":

    st.header("🗂️ Dataset Explorer")

    st.write(
        "Filter records, preview the dataset, and export "
        "matching rows as a CSV file."
    )

    filtered_df = df.copy()

    # --------------------------------------------------------
    # RATING FILTER
    # --------------------------------------------------------

    rating_options = sorted(
        filtered_df["star_rating"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_ratings = st.multiselect(
        "Filter by Star Rating",
        options=rating_options,
        default=rating_options,
    )

    filtered_df = filtered_df[
        filtered_df["star_rating"].isin(selected_ratings)
    ]

    # --------------------------------------------------------
    # SENTIMENT FILTER
    # --------------------------------------------------------

    sentiment_options = sorted(
        filtered_df["sentiment"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_sentiments = st.multiselect(
        "Filter by Sentiment",
        options=sentiment_options,
        default=sentiment_options,
    )

    filtered_df = filtered_df[
        filtered_df["sentiment"]
        .astype(str)
        .isin(selected_sentiments)
    ]

    # --------------------------------------------------------
    # AUTHENTICITY LABEL FILTER
    # --------------------------------------------------------

    label_choice = st.selectbox(
        "Filter by Authenticity Label",
        [
            "All",
            "Real-labelled (0)",
            "Fake-labelled (1)",
        ],
    )

    if label_choice == "Real-labelled (0)":

        filtered_df = filtered_df[
            filtered_df["is_fake_review"] == 0
        ]

    elif label_choice == "Fake-labelled (1)":

        filtered_df = filtered_df[
            filtered_df["is_fake_review"] == 1
        ]

    # --------------------------------------------------------
    # DISPLAY FILTERED RECORDS
    # --------------------------------------------------------

    st.metric(
        "Matching Records",
        f"{len(filtered_df):,}",
    )

    st.dataframe(
        filtered_df.head(1000),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The on-screen preview is limited to 1,000 rows. "
        "The download includes all filtered rows."
    )

    # --------------------------------------------------------
    # DOWNLOAD FILTERED CSV
    # --------------------------------------------------------

    st.download_button(
        label="⬇️ Download Filtered Dataset",
        data=filtered_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="guardrail_filtered_records.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# 6. FOOTER
# ============================================================

st.divider()

st.caption(
    "E-Commerce Guardrail | Research Prototype | "
    "Human review is required before consequential decisions."
)
