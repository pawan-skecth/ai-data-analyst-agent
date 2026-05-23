import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ollama

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Data Analyst Agent",
    layout="wide"
)

MODEL_NAME = "llama3.2:3b"

st.title("📊 AI Data Analyst Agent")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📂 Navigation")

section = st.sidebar.radio(
    "Go To",
    [
        "Overview",
        "EDA",
        "Visualizations",
        "ML Insights",
        "AI Agent"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# =====================================================
# INITIAL VARIABLES
# =====================================================

df = None
preprocessing_report = ""

# =====================================================
# LOAD DATA
# =====================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    rows = df.shape[0]
    cols = df.shape[1]

    # =====================================================
    # CLEAN DATATYPES
    # =====================================================

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_cols = df.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    # =====================================================
    # REMOVE ID COLUMNS FROM ML
    # =====================================================

    id_columns = []

    for col in df.columns:

        unique_ratio = df[col].nunique() / len(df)

        if unique_ratio > 0.95:
            id_columns.append(col)

    # =====================================================
    # MISSING VALUES
    # =====================================================

    missing_values = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values,
        "Missing %": (
            missing_values.values / len(df)
        ) * 100
    })

    # =====================================================
    # MISSING VALUE RECOMMENDATIONS
    # =====================================================

    missing_recommendation = ""

    for col in df.columns:

        missing_percent = (
            df[col].isnull().sum() / len(df)
        ) * 100

        if missing_percent > 40:

            missing_recommendation += (
                f"{col}: Consider dropping column\n"
            )

        elif col in numeric_cols:

            skewness = df[col].skew()

            if abs(skewness) > 1:

                missing_recommendation += (
                    f"{col}: Median imputation recommended\n"
                )

            else:

                missing_recommendation += (
                    f"{col}: Mean imputation recommended\n"
                )

        else:

            missing_recommendation += (
                f"{col}: Mode imputation recommended\n"
            )

    # =====================================================
    # DUPLICATES
    # =====================================================

    duplicates = df.duplicated().sum()

    # =====================================================
    # OUTLIER REPORT
    # =====================================================

    outlier_report = ""

    for col in numeric_cols:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)

        outliers = df[
            (df[col] < lower) |
            (df[col] > upper)
        ].shape[0]

        outlier_report += (
            f"{col}: {outliers} outliers\n"
        )

    if outlier_report == "":
        outlier_report = "No numeric columns found"

    # =====================================================
    # SMART ENCODING ANALYSIS
    # =====================================================

    encoding_report = ""

    ordinal_keywords = [
        "low",
        "medium",
        "high",
        "poor",
        "good",
        "excellent"
    ]

    for col in categorical_cols:

        unique_values = df[col].nunique()

        sample_values = (
            df[col]
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
            .tolist()
        )

        # Binary Encoding
        if unique_values == 2:

            encoding_report += (
                f"{col}: Binary Encoding recommended\n"
            )

        # Ordinal Encoding
        elif any(
            keyword in sample_values
            for keyword in ordinal_keywords
        ):

            encoding_report += (
                f"{col}: Ordinal Encoding recommended\n"
            )

        # One Hot Encoding
        elif unique_values <= 10:

            encoding_report += (
                f"{col}: One Hot Encoding recommended\n"
            )

        # High Cardinality
        else:

            encoding_report += (
                f"{col}: Label Encoding recommended "
                f"(High Cardinality)\n"
            )

    if encoding_report == "":
        encoding_report = "No categorical columns found"

    # =====================================================
    # SCALING REPORT
    # =====================================================

    scaling_report = ""

    for col in numeric_cols:

        if abs(df[col].skew()) > 1:

            scaling_report += (
                f"{col}: Scaling recommended\n"
            )

    if scaling_report == "":
        scaling_report = "No scaling required"

    # =====================================================
    # CORRELATION REPORT
    # =====================================================

    high_corr_report = ""

    if len(numeric_cols) > 1:

        corr_matrix = df[numeric_cols].corr()

        for i in range(len(corr_matrix.columns)):

            for j in range(i):

                corr_value = corr_matrix.iloc[i, j]

                if abs(corr_value) > 0.8:

                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]

                    high_corr_report += (
                        f"{col1} and {col2} "
                        f"highly correlated ({corr_value:.2f})\n"
                    )

    if high_corr_report == "":
        high_corr_report = (
            "No strong correlations found"
        )

    # =====================================================
    # TARGET COLUMN
    # =====================================================

    st.sidebar.subheader("🎯 Target Column")

    target_column = st.sidebar.selectbox(
        "Select Target Column",
        df.columns
    )

    # =====================================================
    # FEATURE IMPORTANCE + MODEL EVALUATION
    # =====================================================

    feature_importance_df = None
    ml_problem_type = ""

    accuracy = None
    precision = None
    recall = None
    f1 = None

    mae = None
    mse = None
    rmse = None
    r2 = None

    conf_matrix = None

    try:

        temp_df = df.copy()

        # Remove ID columns
        for col in id_columns:

            if col != target_column:

                temp_df.drop(
                    columns=[col],
                    inplace=True
                )

        # Fill missing values
        for col in temp_df.columns:

            if col in numeric_cols:

                temp_df[col] = temp_df[col].fillna(
                    temp_df[col].median()
                )

            else:

                temp_df[col] = temp_df[col].fillna(
                    temp_df[col].mode()[0]
                )

        # Encode categorical columns
        le = LabelEncoder()

        for col in categorical_cols:

            if col in temp_df.columns:

                temp_df[col] = le.fit_transform(
                    temp_df[col].astype(str)
                )

        X = temp_df.drop(columns=[target_column])
        y = temp_df[target_column]

        # Smart Target Detection
        unique_target = y.nunique()
        unique_ratio = unique_target / len(y)

        if (
            target_column in categorical_cols
            or unique_ratio < 0.05
            or unique_target <= 15
        ):

            ml_problem_type = "Classification"

            if target_column in categorical_cols:

                y = le.fit_transform(
                    y.astype(str)
                )

            model = RandomForestClassifier(
                random_state=42
            )

        else:

            ml_problem_type = "Regression"

            model = RandomForestRegressor(
                random_state=42
            )

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Train Model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Feature Importance
        importance = model.feature_importances_

        feature_importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        })

        feature_importance_df = (
            feature_importance_df
            .sort_values(
                by="Importance",
                ascending=False
            )
        )

        # Classification Metrics
        if ml_problem_type == "Classification":

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            conf_matrix = confusion_matrix(
                y_test,
                y_pred
            )

        # Regression Metrics
        else:

            mae = mean_absolute_error(
                y_test,
                y_pred
            )

            mse = mean_squared_error(
                y_test,
                y_pred
            )

            rmse = np.sqrt(mse)

            r2 = r2_score(
                y_test,
                y_pred
            )

    except Exception as e:

        st.error(f"ML Error: {e}")

    # =====================================================
    # PREPROCESSING REPORT
    # =====================================================

    preprocessing_report = f"""
Dataset Information:
Rows: {rows}
Columns: {cols}

Numeric Columns:
{numeric_cols}

Categorical Columns:
{categorical_cols}

Detected ID Columns:
{id_columns}

Missing Value Recommendations:
{missing_recommendation}

Duplicate Rows:
{duplicates}

Outlier Report:
{outlier_report}

Encoding Suggestions:
{encoding_report}

Scaling Suggestions:
{scaling_report}

Correlation Insights:
{high_corr_report}

ML Problem Type:
{ml_problem_type}

Selected Target Column:
{target_column}
"""

    # =====================================================
    # OVERVIEW
    # =====================================================

    if section == "Overview":

        st.header("📋 Dataset Overview")

        st.dataframe(df.head())

        st.subheader("📊 Dataset Shape")

        st.write(f"Rows: {rows}")
        st.write(f"Columns: {cols}")

        st.subheader("🧾 Data Types")

        st.dataframe(
            pd.DataFrame(
                df.dtypes,
                columns=["Data Type"]
            )
        )

        st.subheader("🆔 Detected ID Columns")

        st.write(id_columns)

        st.subheader("🎯 Selected Target Column")

        st.success(target_column)

    # =====================================================
    # EDA
    # =====================================================

    elif section == "EDA":

        st.header("📊 Exploratory Data Analysis")

        st.subheader("❗ Missing Values")

        st.dataframe(missing_df)

        st.subheader("💡 Missing Value Suggestions")

        st.text(missing_recommendation)

        st.subheader("🔁 Duplicate Rows")

        st.write(duplicates)

        st.subheader("📦 Outlier Report")

        st.text(outlier_report)

        st.subheader("🏷️ Encoding Suggestions")

        st.text(encoding_report)

        st.subheader("⚖️ Scaling Suggestions")

        st.text(scaling_report)

        st.subheader("🔥 Correlation Insights")

        st.text(high_corr_report)

    # =====================================================
    # VISUALIZATIONS
    # =====================================================

    elif section == "Visualizations":

        st.header("📈 Visualizations")

        chart_type = st.selectbox(
            "Select Chart",
            [
                "Histogram",
                "Boxplot",
                "Bar Chart",
                "Pie Chart",
                "Scatter Plot",
                "Line Chart",
                "Correlation Heatmap"
            ]
        )

        if chart_type == "Histogram":

            selected_column = st.selectbox(
                "Select Numeric Column",
                numeric_cols
            )

            fig, ax = plt.subplots()

            ax.hist(
                df[selected_column].dropna(),
                bins=20
            )

            st.pyplot(fig)

        elif chart_type == "Boxplot":

            selected_column = st.selectbox(
                "Select Numeric Column",
                numeric_cols
            )

            fig, ax = plt.subplots()

            ax.boxplot(
                df[selected_column].dropna()
            )

            st.pyplot(fig)

        elif chart_type == "Bar Chart":

            selected_column = st.selectbox(
                "Select Categorical Column",
                categorical_cols
            )

            value_counts = (
                df[selected_column]
                .value_counts()
                .head(10)
            )

            fig, ax = plt.subplots()

            ax.bar(
                value_counts.index.astype(str),
                value_counts.values
            )

            plt.xticks(rotation=45)

            st.pyplot(fig)

        elif chart_type == "Pie Chart":

            selected_column = st.selectbox(
                "Select Categorical Column",
                categorical_cols
            )

            value_counts = (
                df[selected_column]
                .value_counts()
                .head(5)
            )

            fig, ax = plt.subplots()

            ax.pie(
                value_counts.values,
                labels=value_counts.index.astype(str),
                autopct='%1.1f%%'
            )

            st.pyplot(fig)

        elif chart_type == "Scatter Plot":

            x_col = st.selectbox(
                "Select X Column",
                numeric_cols
            )

            y_col = st.selectbox(
                "Select Y Column",
                numeric_cols
            )

            fig, ax = plt.subplots()

            ax.scatter(
                df[x_col],
                df[y_col]
            )

            st.pyplot(fig)

        elif chart_type == "Line Chart":

            selected_column = st.selectbox(
                "Select Numeric Column",
                numeric_cols
            )

            fig, ax = plt.subplots()

            ax.plot(df[selected_column])

            st.pyplot(fig)

        elif chart_type == "Correlation Heatmap":

            if len(numeric_cols) > 1:

                corr_matrix = (
                    df[numeric_cols]
                    .corr()
                )

                fig, ax = plt.subplots(
                    figsize=(10, 6)
                )

                cax = ax.imshow(
                    corr_matrix,
                    cmap="coolwarm"
                )

                ax.set_xticks(
                    range(len(numeric_cols))
                )

                ax.set_yticks(
                    range(len(numeric_cols))
                )

                ax.set_xticklabels(
                    numeric_cols,
                    rotation=90
                )

                ax.set_yticklabels(
                    numeric_cols
                )

                fig.colorbar(cax)

                st.pyplot(fig)

    # =====================================================
    # ML INSIGHTS
    # =====================================================

    elif section == "ML Insights":

        st.header("🤖 ML Insights")

        st.subheader("🎯 Selected Target Column")

        st.success(target_column)

        st.subheader("🧠 Problem Type")

        st.success(ml_problem_type)

        st.subheader("📊 Target Distribution")

        fig, ax = plt.subplots()

        df[target_column].value_counts().head(10).plot(
            kind="bar",
            ax=ax
        )

        st.pyplot(fig)

        if feature_importance_df is not None:

            st.subheader("⭐ Feature Importance")

            st.dataframe(feature_importance_df)

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.bar(
                feature_importance_df["Feature"],
                feature_importance_df["Importance"]
            )

            plt.xticks(rotation=90)

            st.pyplot(fig)

            # =====================================================
            # MODEL PERFORMANCE
            # =====================================================

            st.subheader("📈 Model Performance")

            # Classification Metrics
            if ml_problem_type == "Classification":

                st.write(f"Accuracy: {accuracy:.4f}")
                st.write(f"Precision: {precision:.4f}")
                st.write(f"Recall: {recall:.4f}")
                st.write(f"F1 Score: {f1:.4f}")

                # Confusion Matrix
                st.subheader("🔥 Confusion Matrix")

                fig, ax = plt.subplots()

                cax = ax.imshow(
                    conf_matrix,
                    cmap="Blues"
                )

                fig.colorbar(cax)

                st.pyplot(fig)

            # Regression Metrics
            else:

                st.write(f"MAE: {mae:.4f}")
                st.write(f"MSE: {mse:.4f}")
                st.write(f"RMSE: {rmse:.4f}")
                st.write(f"R² Score: {r2:.4f}")

    # =====================================================
    # AI AGENT
    # =====================================================

    elif section == "AI Agent":

        st.header("🧠 AI Data Analyst")

        user_input = st.text_area(
            "Ask questions about your dataset"
        )

        if st.button("Run Agent"):

            if user_input.strip() == "":

                st.warning(
                    "Please enter question"
                )

            else:

                try:

                    final_prompt = f"""
You are an expert AI Data Analyst.

Below is a preprocessing and ML report.

{preprocessing_report}

Feature Importance:
{feature_importance_df.head(10).to_string(index=False)
if feature_importance_df is not None else "No feature importance available"}

User Question:
{user_input}

Instructions:
- Give concise answer
- Mention preprocessing suggestions
- Mention ML readiness
- Mention important features
- Mention business insights if possible
"""

                    response = ollama.chat(
                        model=MODEL_NAME,
                        messages=[
                            {
                                "role": "user",
                                "content": final_prompt
                            }
                        ]
                    )

                    st.subheader("🧠 AI Response")

                    st.write(
                        response["message"]["content"]
                    )

                except Exception as e:

                    st.error(f"Error: {e}")

else:

    st.info(
        "Please upload a CSV file from sidebar"
    )