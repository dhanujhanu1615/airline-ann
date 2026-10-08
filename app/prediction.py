#Step 1: Importing the Libraries

import pandas as pd


#Step 2: Creating the Passenger Input DataFrame

def create_input_dataframe(
    gender,
    customer_type,
    age,
    travel_type,
    travel_class,
    flight_distance,
    wifi,
    departure_arrival,
    online_booking,
    gate_location,
    food_drink,
    online_boarding,
    seat_comfort,
    inflight_entertainment,
    onboard_service,
    leg_room,
    baggage,
    checkin,
    inflight_service,
    cleanliness,
    departure_delay,
    arrival_delay
):

    input_data = pd.DataFrame({

        "Gender": [gender],

        "Customer Type": [customer_type],

        "Age": [age],

        "Type of Travel": [travel_type],

        "Class": [travel_class],

        "Flight Distance": [flight_distance],

        "Inflight wifi service": [wifi],

        "Departure/Arrival time convenient": [
            departure_arrival
        ],

        "Ease of Online booking": [
            online_booking
        ],

        "Gate location": [
            gate_location
        ],

        "Food and drink": [
            food_drink
        ],

        "Online boarding": [
            online_boarding
        ],

        "Seat comfort": [
            seat_comfort
        ],

        "Inflight entertainment": [
            inflight_entertainment
        ],

        "On-board service": [
            onboard_service
        ],

        "Leg room service": [
            leg_room
        ],

        "Baggage handling": [
            baggage
        ],

        "Checkin service": [
            checkin
        ],

        "Inflight service": [
            inflight_service
        ],

        "Cleanliness": [
            cleanliness
        ],

        "Departure Delay in Minutes": [
            departure_delay
        ],

        "Arrival Delay in Minutes": [
            arrival_delay
        ]
    })

    return input_data


#Step 3: Generating the ANN Prediction

def predict_satisfaction(
    model,
    preprocessor,
    input_data,
    threshold=0.5
):

    processed_data = preprocessor.transform(
        input_data
    )

    probability = float(
        model.predict(
            processed_data,
            verbose=0
        )[0][0]
    )

    prediction = int(
        probability >= threshold
    )

    confidence = (
        probability
        if prediction == 1
        else 1 - probability
    )

    return {
        "prediction": prediction,
        "probability": probability,
        "confidence": confidence
    }


#Step 4: Calculating Passenger Experience Score

def calculate_experience_score(input_data):

    service_columns = [

        "Inflight wifi service",

        "Departure/Arrival time convenient",

        "Ease of Online booking",

        "Gate location",

        "Food and drink",

        "Online boarding",

        "Seat comfort",

        "Inflight entertainment",

        "On-board service",

        "Leg room service",

        "Baggage handling",

        "Checkin service",

        "Inflight service",

        "Cleanliness"
    ]

    service_average = (
        input_data[
            service_columns
        ]
        .mean(axis=1)
        .iloc[0]
    )

    service_score = (
        service_average / 5
    ) * 100


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

    delay_penalty = min(
        total_delay * 0.15,
        25
    )

    experience_score = max(
        0,
        service_score - delay_penalty
    )

    return round(
        experience_score,
        1
    )


#Step 5: Calculating Service Health

def calculate_service_health(input_data):

    service_mapping = {

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

    for display_name, column in service_mapping.items():

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

    return pd.DataFrame(
        results
    )


#Step 6: Creating Business Recommendations

def generate_recommendations(
    input_data,
    prediction
):

    recommendations = []


    if input_data[
        "Inflight wifi service"
    ].iloc[0] <= 2:

        recommendations.append(
            (
                "WiFi Experience",
                "Improve inflight connectivity "
                "and network reliability."
            )
        )


    if input_data[
        "Online boarding"
    ].iloc[0] <= 2:

        recommendations.append(
            (
                "Digital Boarding",
                "Improve the online boarding journey "
                "and reduce digital friction."
            )
        )


    if input_data[
        "Seat comfort"
    ].iloc[0] <= 2:

        recommendations.append(
            (
                "Passenger Comfort",
                "Review seat comfort and "
                "cabin ergonomics."
            )
        )


    if input_data[
        "Cleanliness"
    ].iloc[0] <= 2:

        recommendations.append(
            (
                "Cabin Cleanliness",
                "Increase cabin cleanliness monitoring."
            )
        )


    if input_data[
        "Inflight entertainment"
    ].iloc[0] <= 2:

        recommendations.append(
            (
                "Entertainment",
                "Improve the availability and quality "
                "of inflight entertainment."
            )
        )


    if input_data[
        "Departure Delay in Minutes"
    ].iloc[0] > 30:

        recommendations.append(
            (
                "Departure Reliability",
                "Investigate recurring departure delays."
            )
        )


    if input_data[
        "Arrival Delay in Minutes"
    ].iloc[0] > 30:

        recommendations.append(
            (
                "Arrival Reliability",
                "Investigate excessive arrival delays."
            )
        )


    if (
        prediction == 1
        and len(recommendations) == 0
    ):

        recommendations.append(
            (
                "Maintain Experience",
                "Current passenger experience appears healthy. "
                "Continue maintaining service quality."
            )
        )


    if (
        prediction == 0
        and len(recommendations) == 0
    ):

        recommendations.append(
            (
                "Experience Review",
                "Review the complete passenger journey "
                "for potential satisfaction drivers."
            )
        )


    return recommendations