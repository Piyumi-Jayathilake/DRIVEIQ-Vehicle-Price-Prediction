
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Vehicle Price Analytics",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
footer {
    visibility: hidden;
}
[data-testid="stFooter"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ==========================================
# LOAD DATA AND MODELS
# ==========================================

@st.cache_data
def load_data():
    data_path = DATA_DIR / "cleaned_used_cars.csv"
    return pd.read_csv(data_path)


@st.cache_resource
def load_model():
    model_path = MODEL_DIR / "stacking_pipeline.pkl"
    return joblib.load(model_path)


@st.cache_data
def load_results():
    results_path = MODEL_DIR / "model_results.pkl"
    return joblib.load(results_path)


@st.cache_data
def load_features():
    features_path = MODEL_DIR / "feature_information.pkl"
    return joblib.load(features_path)


df = load_data()
model = load_model()
results = load_results()
comparison_df = pd.DataFrame(results).T
feature_info = load_features()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("Vehicle Analytics")
st.sidebar.caption("Used Vehicle Price Intelligence")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard Overview",
        "Data Analysis",
        "Model Performance",
        "Price Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "224092J\n"
    "Jayathilake P. P. P."
)

# ==========================================
# DASHBOARD OVERVIEW
# ==========================================

if page == "Dashboard Overview":

    st.title("Vehicle Price Analytics Dashboard")

    st.write(
        "Explore the used vehicle market "
        "and understand vehicle price trends."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Vehicles",
        f"{len(df):,}"
    )

    col2.metric(
        "Vehicle Brands",
        df["Brand"].nunique()
    )

    col3.metric(
        "Average Price",
        f"{df['Price'].mean():.2f}"
    )

    col4.metric(
        "Average Mileage",
        f"{df['Millage(KM)'].mean():,.0f} KM"
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Vehicles by Brand")

        brand_counts = (
            df["Brand"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        brand_counts.columns = ["Brand", "Count"]

        fig = px.bar(
            brand_counts,
            x="Brand",
            y="Count",
            title="Top 10 Vehicle Brands"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("Vehicle Price Distribution")

        fig = px.histogram(
            df,
            x="Price",
            nbins=40,
            title="Distribution of Vehicle Prices"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Recent Dataset Records")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

# ==========================================
# DATA ANALYSIS
# ==========================================

elif page == "Data Analysis":

    st.title("Vehicle Data Analysis")

    st.write(
        "Explore the dataset using interactive visualizations."
    )

    brand_options = ["All"] + sorted(
        df["Brand"].dropna().unique().tolist()
    )

    selected_brand = st.selectbox(
        "Select Vehicle Brand",
        brand_options
    )

    filtered_df = df.copy()

    if selected_brand != "All":

        filtered_df = filtered_df[
            filtered_df["Brand"] == selected_brand
        ]

    col1, col2 = st.columns(2)

    with col1:

        fig = px.box(
            filtered_df,
            x="Brand",
            y="Price",
            title="Vehicle Prices by Brand"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.scatter(
            filtered_df,
            x="Millage(KM)",
            y="Price",
            color="Brand",
            title="Mileage vs Vehicle Price"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col3, col4 = st.columns(2)

    with col3:

        year_prices = (
            filtered_df.groupby("YOM")["Price"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            year_prices,
            x="YOM",
            y="Price",
            markers=True,
            title="Average Price by Manufacturing Year"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col4:

        fuel_prices = (
            filtered_df.groupby("Fuel Type")["Price"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            fuel_prices,
            x="Fuel Type",
            y="Price",
            title="Average Price by Fuel Type"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Filtered Vehicle Records")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

# ==========================================
# MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.title("Ensemble Model Performance")

    st.write(
        "Comparison of trained machine learning models "
        "using evaluation metrics."
    )

    comparison_df = pd.DataFrame(results).T

    comparison_df.index.name = "Model"

    comparison_df = comparison_df.reset_index()

    st.subheader("Model Comparison")

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            comparison_df,
            x="Model",
            y="R2",
            title="R² Score Comparison",
            text_auto=".4f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            comparison_df,
            x="Model",
            y="RMSE",
            title="RMSE Comparison",
            text_auto=".4f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    best_model = comparison_df.loc[
        comparison_df["R2"].idxmax()
    ]

    st.success(
        f"Best model: {best_model['Model']} | "
        f"R² Score: {best_model['R2']:.4f}"
    )

# ==========================================
# PRICE PREDICTION
# ==========================================

elif page == "Price Prediction":

    st.title("Vehicle Price Prediction")

    st.write(
        "Enter the vehicle specifications "
        "to estimate its price using the trained Stacking model."
    )

    feature_columns = feature_info["feature_columns"]

    categorical_columns = feature_info["categorical_columns"]

    numerical_columns = feature_info["numerical_columns"]

    with st.form("prediction_form"):

        st.subheader("Vehicle Specifications")

        user_input = {}

        cols = st.columns(2)

        for i, feature in enumerate(feature_columns):

            with cols[i % 2]:

                if feature in categorical_columns:

                    options = sorted(
                        df[feature]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    user_input[feature] = st.selectbox(
                        feature,
                        options
                    )

                else:

                    minimum = float(df[feature].min())

                    maximum = float(df[feature].max())

                    default = float(df[feature].median())

                    if feature == "YOM":

                        user_input[feature] = st.number_input(
                            feature,
                            min_value=int(minimum),
                            max_value=int(maximum),
                            value=int(default),
                            step=1,
                            format="%d"
                        )

                    else:

                        user_input[feature] = st.number_input(
                            feature,
                            min_value=minimum,
                            max_value=maximum,
                            value=default
                        )

        submitted = st.form_submit_button(
            "Predict Vehicle Price",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame(
            [user_input],
            columns=feature_columns
        )

        prediction = model.predict(input_df)[0]

        st.divider()

        st.subheader("Predicted Vehicle Price")

        st.metric(
            "Estimated Price",
            f"{prediction:,.2f}"
        )

        st.info(
            "This prediction is generated using "
            "the trained Stacking Ensemble model."
        )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Vehicle Price Analytics Dashboard"
)
