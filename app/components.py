#Step 1: Importing the Libraries

import streamlit as st
import pandas as pd


#Step 2: Creating the Main Application Header

def application_header(
    title="AirlineIQ",
    subtitle="Passenger Experience Intelligence Platform"
):

    st.markdown(
        f"# ✈️ {title}"
    )

    st.caption(
        subtitle
    )

    st.divider()


#Step 3: Creating Section Headers

def section_header(
    title,
    subtitle=None
):

    st.markdown(
        f"### {title}"
    )

    if subtitle:

        st.caption(
            subtitle
        )


#Step 4: Creating KPI Cards

def display_kpis(
    accuracy=None,
    precision=None,
    recall=None,
    f1_score=None,
    roc_auc=None,
    **kwargs
):

    metrics = []

    if accuracy is not None:

        metrics.append(
            (
                "Accuracy",
                accuracy
            )
        )

    if precision is not None:

        metrics.append(
            (
                "Precision",
                precision
            )
        )

    if recall is not None:

        metrics.append(
            (
                "Recall",
                recall
            )
        )

    if f1_score is not None:

        metrics.append(
            (
                "F1 Score",
                f1_score
            )
        )

    if roc_auc is not None:

        metrics.append(
            (
                "ROC-AUC",
                roc_auc
            )
        )

    if not metrics:

        return

    columns = st.columns(
        len(metrics)
    )

    for column, (label, value) in zip(
        columns,
        metrics
    ):

        with column:

            if isinstance(
                value,
                (float, int)
            ):

                if 0 <= value <= 1:

                    formatted_value = (
                        f"{value * 100:.1f}%"
                    )

                else:

                    formatted_value = (
                        f"{value:.2f}"
                    )

            else:

                formatted_value = str(
                    value
                )

            st.metric(
                label=label,
                value=formatted_value
            )


#Step 5: Displaying Prediction Result

def display_prediction_result(
    prediction,
    probability,
    confidence
):

    if prediction == 1:

        st.success(
            "## ✅ Passenger Likely Satisfied"
        )

    else:

        st.error(
            "## ⚠️ Passenger Likely Neutral / Dissatisfied"
        )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(
            "Satisfaction Probability",
            f"{probability * 100:.1f}%"
        )

    with col2:

        st.metric(
            "Model Confidence",
            f"{confidence * 100:.1f}%"
        )

    with col3:

        if prediction == 1:

            st.metric(
                "Predicted Class",
                "Satisfied"
            )

        else:

            st.metric(
                "Predicted Class",
                "Neutral / Dissatisfied"
            )

    st.progress(
        min(
            max(
                probability,
                0.0
            ),
            1.0
        )
    )


#Step 6: Displaying Passenger Experience Score

def display_experience_score(
    score
):

    if score >= 80:

        status = "Excellent"
        icon = "🟢"

    elif score >= 60:

        status = "Good"
        icon = "🟡"

    elif score >= 40:

        status = "Needs Improvement"
        icon = "🟠"

    else:

        status = "Poor"
        icon = "🔴"

    col1, col2 = st.columns(
        2
    )

    with col1:

        st.metric(
            "Passenger Experience Score",
            f"{score:.1f}/100"
        )

    with col2:

        st.metric(
            "Experience Health",
            f"{icon} {status}"
        )

    st.progress(
        min(
            max(
                score / 100,
                0.0
            ),
            1.0
        )
    )


#Step 7: Displaying Service Health

def display_service_health(
    service_health
):

    if (
        service_health is None
        or service_health.empty
    ):

        st.info(
            "Service health information is unavailable."
        )

        return

    for _, row in service_health.iterrows():

        service = row[
            "Service"
        ]

        rating = float(
            row[
                "Rating"
            ]
        )

        col1, col2 = st.columns(
            [4, 1]
        )

        with col1:

            st.markdown(
                f"**{service}**"
            )

        with col2:

            st.markdown(
                f"**{rating:.1f}/5**"
            )

        st.progress(
            min(
                max(
                    rating / 5,
                    0.0
                ),
                1.0
            )
        )


#Step 8: Displaying Business Recommendations

def display_recommendations(
    recommendations
):

    if not recommendations:

        st.success(
            "✅ No immediate service recommendations."
        )

        return

    for title, description in recommendations:

        with st.container(
            border=True
        ):

            st.markdown(
                f"**💡 {title}**"
            )

            st.write(
                description
            )


#Step 9: Displaying Model Information

def display_model_information(
    model_name,
    threshold=0.5
):

    col1, col2 = st.columns(
        2
    )

    with col1:

        st.metric(
            "Model",
            model_name
        )

    with col2:

        st.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )


#Step 10: Displaying Prediction Interpretation

def display_prediction_interpretation(
    prediction,
    probability
):

    if prediction == 1:

        st.info(
            f"""
            **🤖 ANN Interpretation**

            The model estimates a **{probability * 100:.1f}% probability**
            that this passenger will be satisfied.

            The satisfaction probability is above the configured
            decision threshold.
            """
        )

    else:

        dissatisfaction_probability = (
            1 - probability
        )

        st.warning(
            f"""
            **🤖 ANN Interpretation**

            The model estimates a **{dissatisfaction_probability * 100:.1f}% probability**
            that this passenger will be neutral or dissatisfied.

            The satisfaction probability is below the configured
            decision threshold.
            """
        )


#Step 11: Displaying Passenger Profile

def display_passenger_profile(
    input_data
):

    if (
        input_data is None
        or input_data.empty
    ):

        st.info(
            "Passenger profile is unavailable."
        )

        return

    row = input_data.iloc[0]

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.markdown(
            f"**Age**  \n{row['Age']:.0f}"
        )

        st.markdown(
            f"**Gender**  \n{row['Gender']}"
        )

    with col2:

        st.markdown(
            f"**Customer Type**  \n{row['Customer Type']}"
        )

        st.markdown(
            f"**Travel Type**  \n{row['Type of Travel']}"
        )

    with col3:

        st.markdown(
            f"**Class**  \n{row['Class']}"
        )

        st.markdown(
            f"**Flight Distance**  \n{row['Flight Distance']:.0f} km"
        )


#Step 12: Displaying Delay Analysis

def display_delay_analysis(
    input_data
):

    if (
        input_data is None
        or input_data.empty
    ):

        st.info(
            "Delay information is unavailable."
        )

        return

    departure_delay = float(
        input_data[
            "Departure Delay in Minutes"
        ].iloc[0]
    )

    arrival_delay = float(
        input_data[
            "Arrival Delay in Minutes"
        ].iloc[0]
    )

    total_delay = (
        departure_delay +
        arrival_delay
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(
            "Departure Delay",
            f"{departure_delay:.0f} min"
        )

    with col2:

        st.metric(
            "Arrival Delay",
            f"{arrival_delay:.0f} min"
        )

    with col3:

        st.metric(
            "Total Delay",
            f"{total_delay:.0f} min"
        )

    if total_delay == 0:

        st.success(
            "✈️ Flight is operating without recorded delay."
        )

    elif total_delay <= 30:

        st.info(
            "🟡 Minor delay detected."
        )

    else:

        st.warning(
            "🔴 Significant delay detected."
        )


#Step 13: Displaying Service Rating Summary

def display_service_rating_summary(
    input_data
):

    service_columns = {

        "WiFi":
            "Inflight wifi service",

        "Online Booking":
            "Ease of Online booking",

        "Boarding":
            "Online boarding",

        "Seat Comfort":
            "Seat comfort",

        "Entertainment":
            "Inflight entertainment",

        "Food & Drink":
            "Food and drink",

        "Cleanliness":
            "Cleanliness",

        "Baggage":
            "Baggage handling"
    }

    results = []

    for display_name, column in service_columns.items():

        rating = float(
            input_data[
                column
            ].iloc[0]
        )

        results.append(
            {
                "Service": display_name,
                "Rating": rating
            }
        )

    service_df = pd.DataFrame(
        results
    )

    st.dataframe(
        service_df,
        use_container_width=True,
        hide_index=True
    )


#Step 14: Displaying Detailed Model Information

def display_model_details(
    model=None
):

    st.markdown(
        "### 🧠 ANN Architecture"
    )

    st.info(
        """
        **Input Features → Dense Layers → Dropout Regularization → Output Layer**

        The model uses ReLU activation in hidden layers and a Sigmoid
        activation in the output layer for binary passenger satisfaction
        classification.
        """
    )

    col1, col2 = st.columns(
        2
    )

    with col1:

        st.markdown(
            """
            **Classification**

            `0` → Neutral / Dissatisfied

            `1` → Satisfied
            """
        )

    with col2:

        st.markdown(
            """
            **Framework**

            TensorFlow

            Keras
            """
        )

    if model is not None:

        with st.expander(
            "🔎 View Complete Model Summary"
        ):

            try:

                summary_lines = []

                model.summary(
                    print_fn=summary_lines.append
                )

                st.code(
                    "\n".join(
                        summary_lines
                    )
                )

            except Exception:

                st.info(
                    "Model summary is unavailable."
                )


#Step 15: Displaying Footer

def display_footer():

    st.divider()

    st.caption(
        "✈️ AirlineIQ • Passenger Satisfaction Intelligence Platform"
    )

    st.caption(
        "Artificial Neural Network • TensorFlow / Keras • Streamlit"
    )

    st.caption(
        "Production-oriented Data Science Portfolio Project"
    )