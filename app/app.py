#Step 1: Importing the Libraries

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


#Step 2: Importing Project Configuration

from config import (
    PROJECT_NAME,
    MODEL_NAME,
    TARGET_NAME,
    OPTIMIZED_MODEL_PATH,
    PREPROCESSOR_PATH,
    FEATURE_NAMES_PATH,
    EVALUATION_PATH,
    THRESHOLD_ANALYSIS_PATH,
    PREDICTIONS_PATH,
    OPTIMIZATION_RESULTS_PATH,
    BASELINE_VS_OPTIMIZED_PATH,
    PERMUTATION_IMPORTANCE_PATH,
    TOP_FEATURES_PATH,
    PASSENGER_EXPLANATIONS_PATH,
    BUSINESS_RECOMMENDATIONS_PATH,
    PREDICTION_THRESHOLD
)


#Step 3: Importing UI Components

from components import (
    application_header,
    section_header,
    display_kpis,
    display_prediction_result,
    display_experience_score,
    display_service_health,
    display_recommendations,
    display_model_information,
    display_prediction_interpretation,
    display_passenger_profile,
    display_delay_analysis,
    display_service_rating_summary,
    display_model_details,
    display_footer
)


#Step 4: Importing Prediction Functions

from prediction import (
    create_input_dataframe,
    predict_satisfaction,
    calculate_experience_score,
    calculate_service_health,
    generate_recommendations
)


#Step 5: Configuring the Streamlit Application

st.set_page_config(
    page_title="AirlineIQ",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


#Step 6: Applying Premium Application Styling

st.markdown(
    """
    <style>

    /* Main application */

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }


    /* Sidebar width */

    section[data-testid="stSidebar"] {
        width: 340px !important;
    }


    /* Sidebar content */

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }


    /* Sidebar headings */

    section[data-testid="stSidebar"] h2 {
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.25rem !important;
    }


    section[data-testid="stSidebar"] h3 {
        font-size: 1rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.04rem !important;
    }


    /* Sidebar text */

    section[data-testid="stSidebar"] p {
        font-size: 0.95rem !important;
    }


    /* Navigation radio labels */

    section[data-testid="stSidebar"]
    div[data-testid="stRadio"]
    label {

        font-size: 1.08rem !important;
        font-weight: 700 !important;
        line-height: 1.45 !important;
        padding-top: 0.35rem !important;
        padding-bottom: 0.35rem !important;
    }


    /* Sidebar radio spacing */

    section[data-testid="stSidebar"]
    div[data-testid="stRadio"] > div {

        gap: 0.25rem !important;
    }


    /* Sidebar divider */

    section[data-testid="stSidebar"] hr {

        margin-top: 1.2rem !important;
        margin-bottom: 1.2rem !important;
    }


    /* Metric cards */

    div[data-testid="stMetric"] {

        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 14px;
        padding: 1rem;
    }


    /* Buttons */

    div.stButton > button {

        min-height: 3rem;
        font-size: 1rem;
        font-weight: 700;
        border-radius: 12px;
    }


    /* Tabs */

    button[data-baseweb="tab"] {

        font-weight: 700;
    }


    /* Dataframes */

    div[data-testid="stDataFrame"] {

        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


#Step 7: Creating Safe CSV Loader

def load_csv(
    path
):

    try:

        path = Path(
            path
        )

        if not path.exists():

            return pd.DataFrame()

        return pd.read_csv(
            path
        )

    except Exception:

        return pd.DataFrame()


#Step 8: Loading the ANN Model

@st.cache_resource
def load_model():

    try:

        model_path = Path(
            OPTIMIZED_MODEL_PATH
        )

        if not model_path.exists():

            return None

        return tf.keras.models.load_model(
            model_path
        )

    except Exception as error:

        st.error(
            f"Unable to load ANN model: {error}"
        )

        return None


#Step 9: Loading the Preprocessing Pipeline

@st.cache_resource
def load_preprocessor():

    try:

        preprocessor_path = Path(
            PREPROCESSOR_PATH
        )

        if not preprocessor_path.exists():

            return None

        return joblib.load(
            preprocessor_path
        )

    except Exception as error:

        st.error(
            f"Unable to load preprocessing pipeline: {error}"
        )

        return None


#Step 10: Loading Feature Names

@st.cache_data
def load_feature_names():

    try:

        feature_path = Path(
            FEATURE_NAMES_PATH
        )

        if not feature_path.exists():

            return np.array([])

        return np.load(
            feature_path,
            allow_pickle=True
        )

    except Exception:

        return np.array([])


#Step 11: Loading All Analysis Results

@st.cache_data
def load_project_data():

    return {

        "evaluation":
            load_csv(
                EVALUATION_PATH
            ),

        "threshold":
            load_csv(
                THRESHOLD_ANALYSIS_PATH
            ),

        "predictions":
            load_csv(
                PREDICTIONS_PATH
            ),

        "optimization":
            load_csv(
                OPTIMIZATION_RESULTS_PATH
            ),

        "baseline":
            load_csv(
                BASELINE_VS_OPTIMIZED_PATH
            ),

        "importance":
            load_csv(
                PERMUTATION_IMPORTANCE_PATH
            ),

        "top_features":
            load_csv(
                TOP_FEATURES_PATH
            ),

        "explanations":
            load_csv(
                PASSENGER_EXPLANATIONS_PATH
            ),

        "recommendations":
            load_csv(
                BUSINESS_RECOMMENDATIONS_PATH
            )
    }


#Step 12: Loading the Application Resources

model = load_model()

preprocessor = load_preprocessor()

feature_names = load_feature_names()

project_data = load_project_data()


evaluation_df = project_data[
    "evaluation"
]

threshold_df = project_data[
    "threshold"
]

predictions_df = project_data[
    "predictions"
]

optimization_df = project_data[
    "optimization"
]

baseline_df = project_data[
    "baseline"
]

importance_df = project_data[
    "importance"
]

top_features_df = project_data[
    "top_features"
]

explanations_df = project_data[
    "explanations"
]

recommendations_df = project_data[
    "recommendations"
]


#Step 13: Creating the Premium Sidebar

with st.sidebar:

    st.markdown(
        "## ✈️ AirlineIQ"
    )

    st.markdown(
        "**Passenger Experience**  \n"
        "**Intelligence Platform**"
    )

    st.divider()

    st.markdown(
        "### 🧭 NAVIGATION"
    )

    st.caption(
        "Explore the AirlineIQ intelligence platform."
    )

    page = st.radio(
        "Navigation",
        [
            "🏠  Dashboard",
            "🤖  AI Predictor",
            "⭐  Experience Analyzer",
            "📈  Model Performance",
            "🔍  Feature Intelligence",
            "ℹ️  About"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        "### 🧠 AI ENGINE"
    )

    st.markdown(
        "**Artificial Neural Network**"
    )

    st.caption(
        "TensorFlow / Keras"
    )

    st.caption(
        "Production ML Application"
    )

    st.divider()

    if (
        model is not None
        and preprocessor is not None
    ):

        st.success(
            "● AI System Ready"
        )

    else:

        st.warning(
            "● Model Files Missing"
        )

    st.caption(
        "AirlineIQ • Passenger Intelligence"
    )


#Step 14: Dashboard Page

if page == "🏠  Dashboard":

    application_header(
        title="AirlineIQ Dashboard",
        subtitle=(
            "Passenger Satisfaction Intelligence "
            "powered by Artificial Neural Networks"
        )
    )

    section_header(
        "Executive Overview",
        "Model performance and passenger intelligence at a glance."
    )


    #Step 14.1: Extracting Evaluation Metrics

    accuracy = None

    precision = None

    recall = None

    f1_score = None

    roc_auc = None


    if not evaluation_df.empty:

        for column in evaluation_df.columns:

            column_name = (
                str(column)
                .lower()
                .replace(
                    " ",
                    "_"
                )
                .replace(
                    "-",
                    "_"
                )
            )

            try:

                value = float(
                    evaluation_df[
                        column
                    ].iloc[0]
                )

            except Exception:

                continue


            if (
                "accuracy"
                in column_name
            ):

                accuracy = value

            elif (
                "precision"
                in column_name
                and "weighted"
                not in column_name
            ):

                precision = value

            elif (
                "recall"
                in column_name
                and "weighted"
                not in column_name
            ):

                recall = value

            elif (
                "f1"
                in column_name
                and "weighted"
                not in column_name
            ):

                f1_score = value

            elif (
                "roc"
                in column_name
                or "auc"
                in column_name
            ):

                roc_auc = value


    #Step 14.2: Displaying Performance KPIs

    display_kpis(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        roc_auc=roc_auc
    )


    st.divider()


    #Step 14.3: Platform Overview

    section_header(
        "AirlineIQ Intelligence Platform",
        "An end-to-end machine learning application."
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(
            "ML Problem",
            "Binary Classification"
        )

        st.caption(
            "Passenger satisfaction prediction"
        )

    with col2:

        st.metric(
            "AI Model",
            "Artificial Neural Network"
        )

        st.caption(
            "TensorFlow / Keras"
        )

    with col3:

        st.metric(
            "Deployment",
            "Streamlit"
        )

        st.caption(
            "Interactive ML application"
        )


    st.divider()


    #Step 14.4: AI System Health

    section_header(
        "AI Engine Health"
    )

    health_col1, health_col2 = st.columns(
        2
    )

    with health_col1:

        if model is not None:

            st.success(
                "🟢 ANN model loaded successfully."
            )

        else:

            st.error(
                "🔴 ANN model unavailable."
            )

    with health_col2:

        if preprocessor is not None:

            st.success(
                "🟢 Preprocessing pipeline loaded."
            )

        else:

            st.error(
                "🔴 Preprocessing pipeline unavailable."
            )


    st.divider()


    #Step 14.5: Business Objective

    section_header(
        "Business Objective"
    )

    st.info(
        """
        AirlineIQ predicts whether a passenger is likely to be satisfied
        using demographic information, travel characteristics, service
        ratings and flight delay information.

        The prediction is then translated into passenger-level insights
        and actionable service recommendations.
        """
    )


#Step 15: AI Predictor Page

elif page == "🤖  AI Predictor":

    application_header(
        title="AI Passenger Predictor",
        subtitle=(
            "Use the trained ANN to estimate passenger satisfaction."
        )
    )

    if (
        model is None
        or preprocessor is None
    ):

        st.error(
            "The ANN model or preprocessing pipeline is unavailable."
        )

        st.stop()


    #Step 15.1: Passenger Profile

    section_header(
        "Passenger Profile",
        "Enter passenger and journey characteristics."
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )

        customer_type = st.selectbox(
            "Customer Type",
            [
                "Loyal Customer",
                "disloyal Customer"
            ]
        )

        age = st.slider(
            "Age",
            min_value=7,
            max_value=85,
            value=35
        )

    with col2:

        travel_type = st.selectbox(
            "Type of Travel",
            [
                "Business travel",
                "Personal Travel"
            ]
        )

        travel_class = st.selectbox(
            "Class",
            [
                "Business",
                "Eco",
                "Eco Plus"
            ]
        )

        flight_distance = st.number_input(
            "Flight Distance",
            min_value=31,
            max_value=4983,
            value=1000
        )

    with col3:

        departure_delay = st.number_input(
            "Departure Delay in Minutes",
            min_value=0,
            max_value=1592,
            value=0
        )

        arrival_delay = st.number_input(
            "Arrival Delay in Minutes",
            min_value=0,
            max_value=1584,
            value=0
        )


    st.divider()


    #Step 15.2: Service Ratings

    section_header(
        "Passenger Service Ratings",
        "Use the dataset rating scale to describe the passenger experience."
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    with col1:

        wifi = st.slider(
            "Inflight WiFi",
            0,
            5,
            3
        )

        departure_arrival = st.slider(
            "Departure / Arrival Convenience",
            0,
            5,
            3
        )

        online_booking = st.slider(
            "Online Booking",
            0,
            5,
            3
        )

        gate_location = st.slider(
            "Gate Location",
            0,
            5,
            3
        )

    with col2:

        food_drink = st.slider(
            "Food & Drink",
            0,
            5,
            3
        )

        online_boarding = st.slider(
            "Online Boarding",
            0,
            5,
            3
        )

        seat_comfort = st.slider(
            "Seat Comfort",
            0,
            5,
            3
        )

        inflight_entertainment = st.slider(
            "Inflight Entertainment",
            0,
            5,
            3
        )

    with col3:

        onboard_service = st.slider(
            "On-board Service",
            0,
            5,
            3
        )

        leg_room = st.slider(
            "Leg Room Service",
            0,
            5,
            3
        )

        baggage = st.slider(
            "Baggage Handling",
            0,
            5,
            3
        )

        checkin = st.slider(
            "Check-in Service",
            0,
            5,
            3
        )

    with col4:

        inflight_service = st.slider(
            "Inflight Service",
            0,
            5,
            3
        )

        cleanliness = st.slider(
            "Cleanliness",
            0,
            5,
            3
        )


    st.divider()


    #Step 15.3: Prediction Button

    predict_button = st.button(
        "🚀  Predict Passenger Satisfaction",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        input_data = create_input_dataframe(

            gender=gender,

            customer_type=customer_type,

            age=age,

            travel_type=travel_type,

            travel_class=travel_class,

            flight_distance=flight_distance,

            wifi=wifi,

            departure_arrival=departure_arrival,

            online_booking=online_booking,

            gate_location=gate_location,

            food_drink=food_drink,

            online_boarding=online_boarding,

            seat_comfort=seat_comfort,

            inflight_entertainment=inflight_entertainment,

            onboard_service=onboard_service,

            leg_room=leg_room,

            baggage=baggage,

            checkin=checkin,

            inflight_service=inflight_service,

            cleanliness=cleanliness,

            departure_delay=departure_delay,

            arrival_delay=arrival_delay
        )


        try:

            result = predict_satisfaction(
                model=model,
                preprocessor=preprocessor,
                input_data=input_data,
                threshold=PREDICTION_THRESHOLD
            )

            prediction = result[
                "prediction"
            ]

            probability = result[
                "probability"
            ]

            confidence = result[
                "confidence"
            ]


            st.divider()


            #Step 15.4: Prediction Result

            section_header(
                "ANN Prediction Result"
            )

            display_prediction_result(
                prediction=prediction,
                probability=probability,
                confidence=confidence
            )


            display_prediction_interpretation(
                prediction=prediction,
                probability=probability
            )


            st.divider()


            #Step 15.5: Passenger Experience Score

            section_header(
                "Passenger Experience Score"
            )

            experience_score = (
                calculate_experience_score(
                    input_data
                )
            )

            display_experience_score(
                experience_score
            )


            st.divider()


            #Step 15.6: Passenger Profile

            section_header(
                "Passenger Profile"
            )

            display_passenger_profile(
                input_data
            )


            st.divider()


            #Step 15.7: Delay Analysis

            section_header(
                "Flight Delay Analysis"
            )

            display_delay_analysis(
                input_data
            )


            st.divider()


            #Step 15.8: Service Health

            section_header(
                "Service Health"
            )

            service_health = (
                calculate_service_health(
                    input_data
                )
            )

            display_service_health(
                service_health
            )


            st.divider()


            #Step 15.9: Business Recommendations

            section_header(
                "Business Recommendations"
            )

            recommendations = (
                generate_recommendations(
                    input_data=input_data,
                    prediction=prediction
                )
            )

            display_recommendations(
                recommendations
            )


        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )

            with st.expander(
                "Technical Error Details"
            ):

                st.exception(
                    error
                )


#Step 16: Experience Analyzer Page

elif page == "⭐  Experience Analyzer":

    application_header(
        title="Passenger Experience Analyzer",
        subtitle=(
            "Understand service quality, delays and "
            "overall passenger experience."
        )
    )

    section_header(
        "How AirlineIQ Measures Experience"
    )

    col1, col2 = st.columns(
        2
    )

    with col1:

        st.info(
            """
            **⭐ Experience Score**

            A business-oriented 0–100 score based on passenger
            service ratings and flight delay impact.
            """
        )

    with col2:

        st.info(
            """
            **🛫 Service Health**

            Evaluates important passenger touchpoints including
            WiFi, boarding, seat comfort, entertainment,
            cleanliness and baggage handling.
            """
        )


    st.divider()


    section_header(
        "Analyze a Passenger"
    )

    st.write(
        """
        Go to **AI Predictor**, enter passenger information and
        click **Predict Passenger Satisfaction**.

        AirlineIQ will automatically generate the experience score,
        service health analysis, delay analysis and recommendations.
        """
    )


#Step 17: Model Performance Page

elif page == "📈  Model Performance":

    application_header(
        title="Model Performance",
        subtitle=(
            "Evaluate the optimized ANN and compare model experiments."
        )
    )


    #Step 17.1: Evaluation Metrics

    section_header(
        "Evaluation Metrics",
        "Final model performance on the test dataset."
    )

    if evaluation_df.empty:

        st.warning(
            "Evaluation summary is unavailable."
        )

    else:

        st.dataframe(
            evaluation_df,
            use_container_width=True,
            hide_index=True
        )


    st.divider()


    #Step 17.2: Threshold Analysis

    section_header(
        "Threshold Analysis",
        "Explore how different classification thresholds affect performance."
    )

    if threshold_df.empty:

        st.info(
            "Threshold analysis is unavailable."
        )

    else:

        st.dataframe(
            threshold_df,
            use_container_width=True,
            hide_index=True
        )


    st.divider()


    #Step 17.3: Optimization Experiments

    section_header(
        "ANN Optimization Experiments"
    )

    if optimization_df.empty:

        st.info(
            "Optimization experiment results are unavailable."
        )

    else:

        st.dataframe(
            optimization_df,
            use_container_width=True,
            hide_index=True
        )


    st.divider()


    #Step 17.4: Baseline Comparison

    section_header(
        "Baseline vs Optimized Model"
    )

    if baseline_df.empty:

        st.info(
            "Baseline comparison is unavailable."
        )

    else:

        st.dataframe(
            baseline_df,
            use_container_width=True,
            hide_index=True
        )


    st.divider()


    #Step 17.5: Model Architecture

    section_header(
        "ANN Model Architecture"
    )

    display_model_information(
        model_name=MODEL_NAME,
        threshold=PREDICTION_THRESHOLD
    )

    display_model_details(
        model=model
    )


#Step 18: Feature Intelligence Page

elif page == "🔍  Feature Intelligence":

    application_header(
        title="Feature Intelligence",
        subtitle=(
            "Discover which passenger attributes matter most "
            "to the ANN predictions."
        )
    )


    #Step 18.1: Permutation Importance

    section_header(
        "Permutation Importance",
        "Higher values indicate greater influence on model performance."
    )

    if importance_df.empty:

        st.warning(
            "Permutation importance data is unavailable."
        )

    else:

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )


        numeric_columns = (
            importance_df.select_dtypes(
                include=np.number
            ).columns
        )


        if len(
            numeric_columns
        ) > 0:

            importance_column = (
                numeric_columns[-1]
            )

            try:

                chart_df = (
                    importance_df.copy()
                )

                feature_column = (
                    chart_df.columns[0]
                )

                chart_df = chart_df.sort_values(
                    importance_column,
                    ascending=True
                )

                st.bar_chart(
                    chart_df.set_index(
                        feature_column
                    )[
                        importance_column
                    ]
                )

            except Exception:

                pass


    st.divider()


    #Step 18.2: Top Features

    section_header(
        "Top Passenger Experience Features"
    )

    if top_features_df.empty:

        st.info(
            "Top feature information is unavailable."
        )

    else:

        st.dataframe(
            top_features_df,
            use_container_width=True,
            hide_index=True
        )


    st.divider()


    #Step 18.3: Passenger Explanations

    section_header(
        "Passenger-Level Explanations"
    )

    if explanations_df.empty:

        st.info(
            "Passenger explanation data is unavailable."
        )

    else:

        st.dataframe(
            explanations_df.head(
                50
            ),
            use_container_width=True,
            hide_index=True
        )


#Step 19: About Page

elif page == "ℹ️  About":

    application_header(
        title="About AirlineIQ",
        subtitle=(
            "An end-to-end Artificial Neural Network "
            "project for airline passenger satisfaction."
        )
    )


    section_header(
        "Project Overview"
    )

    st.write(
        """
        **AirlineIQ** is a real-world Data Science portfolio project
        designed to predict airline passenger satisfaction using an
        Artificial Neural Network.

        The project covers the complete machine learning lifecycle,
        from data understanding and preprocessing to model optimization,
        explainability, business insights and interactive deployment.
        """
    )


    st.divider()


    section_header(
        "Technology Stack"
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.markdown(
            """
            **📊 Data Science**

            • Python  
            • Pandas  
            • NumPy  
            • Scikit-learn
            """
        )

    with col2:

        st.markdown(
            """
            **🧠 Deep Learning**

            • TensorFlow  
            • Keras  
            • Artificial Neural Network
            """
        )

    with col3:

        st.markdown(
            """
            **🚀 Deployment**

            • Streamlit  
            • Joblib  
            • Production preprocessing
            """
        )


    st.divider()


    section_header(
        "Project Pipeline"
    )

    st.code(
        """
Raw Dataset
     ↓
Data Understanding
     ↓
Exploratory Data Analysis
     ↓
Data Preprocessing
     ↓
ANN Model Development
     ↓
Model Evaluation
     ↓
Hyperparameter Optimization
     ↓
Model Explainability
     ↓
Business Insights
     ↓
Streamlit Deployment
        """,
        language="text"
    )


    st.divider()


    section_header(
        "Prediction Configuration"
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(
            "Model",
            "Optimized ANN"
        )

    with col2:

        st.metric(
            "Target",
            TARGET_NAME
        )

    with col3:

        st.metric(
            "Threshold",
            f"{PREDICTION_THRESHOLD:.2f}"
        )


    st.info(
        """
        **Class 0:** Neutral / Dissatisfied

        **Class 1:** Satisfied
        """
    )


#Step 20: Displaying Application Footer

display_footer()