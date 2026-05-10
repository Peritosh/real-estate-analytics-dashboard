import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# ----------------------------------
# PAGE CONFIG
st.set_page_config(
    page_title="Real Estate Analytics",
    layout="wide"
)

# LOAD DATA
@st.cache_data

def load_data():
    
    data = pd.read_csv("Cleaned.zip")

    # Create amenities_count if not present
    if "amenities_count" not in data.columns:
        data["amenities_count"] = (
            data["amenities"].str.count(",") + 1
        )

    return data


data = load_data()

# SIDEBAR NAVIGATION
st.sidebar.title("🏠 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Dataset Overview",
        "EDA Dashboard",
        "Investment Insights"
    ]
)

# HOME PAGE
if page == "Home":

    st.title("🏠 Real Estate Analytics & Investment Insight System")

    st.markdown("""
    ## Features

    - Interactive Real Estate Dashboard
    - Property Filtering System
    - Investment Insights
    - Future Price Estimation
    - Location-wise Analysis
    - EDA Visualizations
    """)

    st.image("banner_image.jpg",width=1000)

#DATASET OVERVIEW
elif page == "Dataset Overview":

    st.title("📊 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", data.shape[0])

    with col2:
        st.metric("Columns", data.shape[1])

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    st.subheader("Column Names")
    st.write(data.columns.tolist())

    st.subheader("Data Types")
    st.write(data.dtypes)

    st.subheader("Missing Values")
    st.write(data.isnull().sum())

# EDA DASHBOARD
elif page == "EDA Dashboard":

    st.title("📈 Exploratory Data Analysis")


# Price Distribution
    st.subheader("Distribution of Property Prices")

    fig1 = px.histogram(
        data,
        x="price_in_lakhs",
        nbins=50,
        color_discrete_sequence=["#1f77b4"]
    )

    st.plotly_chart(fig1, use_container_width=True)

# Property Type vs Price/SqFt
    st.subheader("Price per SqFt by Property Type")

    fig2 = px.box(
        data,
        x="property_type",
        y="price_per_sqft_in_thousands",
        color="property_type"
    )

    st.plotly_chart(fig2, use_container_width=True)


# Size vs Price
    st.subheader("Relationship Between Size & Price")

    fig3 = px.scatter(
        data,
        x="size_in_sqft",
        y="price_in_lakhs",
        color="property_type",
        hover_data=["city", "bhk"],
        opacity=0.5,
        render_mode="webgl"
    )

    fig3.update_traces(
    marker=dict(size=3)
    )

    
    st.plotly_chart(fig3, use_container_width=True)

# Top Cities by Average Price
    st.subheader("Top Cities by Average Property Price")

    city_price = (
        data.groupby("city")["price_in_lakhs"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig4 = px.bar(
        city_price,
        x="city",
        y="price_in_lakhs",
        color="city"
    )

    st.plotly_chart(fig4, use_container_width=True)

# Amenities Impact

    st.subheader("Amenities Impact on Price")

    fig5 = px.box(
        data,
        x="amenities_count",
        y="price_per_sqft_in_thousands",
        color="amenities_count"
    )

    st.plotly_chart(fig5, use_container_width=True)


# Correlation Heatmap
    st.subheader("Correlation Heatmap")

    numeric_data = data.select_dtypes(include=np.number)

    corr = numeric_data.corr()

    fig6, ax = plt.subplots(figsize=(12, 7))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig6)


# INVESTMENT INSIGHTS
elif page == "Investment Insights":

    st.title("💰 Property Investment Insights")

    st.sidebar.subheader("Filter Properties")

# FILTERS
    selected_city = st.sidebar.selectbox(
        "Select City",
        sorted(data["city"].unique())
    )

    selected_bhk = st.sidebar.slider(
        "Select BHK",
        int(data["bhk"].min()),
        int(data["bhk"].max()),
        2
    )

    max_price = st.sidebar.slider(
        "Maximum Budget (Lakhs)",
        int(data["price_in_lakhs"].min()),
        int(data["price_in_lakhs"].max()),
        200
    )

    min_area = st.sidebar.slider(
        "Minimum Area (SqFt)",
        int(data["size_in_sqft"].min()),
        int(data["size_in_sqft"].max()),
        1000
    )

    appreciation_rate = st.sidebar.slider(
        "Expected Annual Appreciation (%)",
        1,
        20,
        8
    ) / 100

# FILTER DATA
    filtered_data = data[
        (data["city"] == selected_city) &
        (data["bhk"] == selected_bhk) &
        (data["price_in_lakhs"] <= max_price) &
        (data["size_in_sqft"] >= min_area)
    ]


# EMPTY CHECK
    if filtered_data.empty:
        st.warning("No properties found for selected filters")

    else:

# KPI METRICS
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Properties Found",
            len(filtered_data)
        )

        col2.metric(
            "Avg Price",
            f"₹ {filtered_data['price_in_lakhs'].mean():.2f} L"
        )

        col3.metric(
            "Avg Price/SqFt",
            f"{filtered_data['price_per_sqft_in_thousands'].mean():.2f} K"
        )

# PROPERTY TABLE
        st.subheader("Filtered Properties")

        st.dataframe(filtered_data)

# INVESTMENT CLASSIFICATION
        avg_price_sqft = filtered_data[
            "price_per_sqft_in_thousands"
        ].mean()

        avg_amenities = filtered_data[
            "amenities_count"
        ].mean()

        if avg_price_sqft < 8 and avg_amenities >= 3:
            investment_result = "✅ Good Investment"

        elif avg_price_sqft < 12:
            investment_result = "⚠ Moderate Investment"

        else:
            investment_result = "❌ Expensive / High Risk"

        st.subheader("Investment Classification")
        st.success(investment_result)

# FUTURE PRICE ESTIMATION
        current_avg_price = filtered_data[
            "price_in_lakhs"
        ].mean()

        future_price = (
            current_avg_price * ((1 + appreciation_rate) ** 5)
        )

        st.subheader("Estimated Average Price After 5 Years")

        st.info(f"₹ {future_price:.2f} Lakhs")

# VISUAL INSIGHTS
        st.subheader("Price Distribution")

        fig7 = px.histogram(
            filtered_data,
            x="price_in_lakhs",
            nbins=30,
            color="property_type"
        )

        st.plotly_chart(fig7, use_container_width=True)

# TOP LOCALITIES
        st.subheader("Top Expensive Localities")

        top_localities = (
            data.groupby("locality")[
                "price_per_sqft_in_thousands"
            ]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig8 = px.bar(
            top_localities,
            x="locality",
            y="price_per_sqft_in_thousands",
            color="locality"
        )

        st.plotly_chart(fig8, use_container_width=True)

# DOWNLOAD BUTTON
        csv = filtered_data.to_csv(index=False)

        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name="filtered_properties.csv",
            mime="text/csv"
        )
