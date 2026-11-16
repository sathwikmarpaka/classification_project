
import streamlit as st
import requests

# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Brake Service Alert",
    page_icon="",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

st.title("🚗 Brake Service Alert Prediction")
st.write(
    "Enter vehicle information to predict whether "
    "a brake service alert is required within 90 days."
)

# --------------------------------
# 2. Vehicle Information Form
# --------------------------------

with st.form("vehicle_prediction_form"):

    st.subheader("Vehicle Information")

    col1, col2 = st.columns(2)

    with col1:
        vehicle_model = st.text_input("Vehicle Model")
        purchase_year = st.number_input(
            "Purchase Year", min_value=1900, max_value=2100, value=2022
        )
        vehicle_age_years = st.number_input(
            "Vehicle Age (Years)", min_value=0, value=4
        )
        under_warranty = st.selectbox(
            "Under Warranty", [0, 1]
        )
        branch_location = st.text_input("Branch Location")
        climate_region = st.text_input("Climate Region")
        financing_type = st.text_input("Financing Type")
        customer_type = st.text_input("Customer Type")
        self_reported_driving_type = st.text_input(
            "Driving Type"
        )
        self_reported_terrain = st.text_input("Terrain")

    with col2:
        number_of_service_visits_total = st.number_input(
            "Total Service Visits", min_value=0, value=3
        )
        current_odometer_km = st.number_input(
            "Current Odometer (KM)", min_value=0, value=40000
        )
        km_since_last_service = st.number_input(
            "KM Since Last Service", min_value=0, value=5000
        )
        days_since_last_service = st.number_input(
            "Days Since Last Service", min_value=0.0, value=180.0
        )
        pad_material = st.text_input("Pad Material")
        km_on_current_pads = st.number_input(
            "KM on Current Pads", min_value=0, value=15000
        )
        pad_thickness_mm = st.number_input(
            "Pad Thickness (MM)", min_value=0.0, value=6.0
        )

        rotor_available = st.checkbox(
            "Rotor Condition Score Available", value=True
        )

        rotor_condition_score = None

        if rotor_available:
            rotor_condition_score = st.number_input(
                "Rotor Condition Score",
                min_value=0.0,
                value=0.7
            )

        brake_fluid_moisture_pct = st.number_input(
            "Brake Fluid Moisture (%)",
            min_value=0.0,
            value=2.0
        )

    st.subheader("Service and Customer Information")

    col3, col4 = st.columns(2)

    with col3:
        last_service_type = st.text_input("Last Service Type")
        last_service_cost_usd = st.number_input(
            "Last Service Cost (USD)", min_value=0.0, value=150.0
        )
        warranty_covered = st.selectbox(
            "Warranty Covered", [0, 1]
        )
        avg_annual_service_spend_usd = st.number_input(
            "Average Annual Service Spend (USD)",
            min_value=0.0,
            value=300.0
        )

    with col4:
        preferred_contact_channel = st.text_input(
            "Preferred Contact Channel"
        )
        previous_outreach_sent = st.number_input(
            "Previous Outreach Sent", min_value=0, value=1
        )
        previous_outreach_response = st.text_input(
            "Previous Outreach Response"
        )

    # --------------------------------
    # 3. Submit Button
    # --------------------------------

    submitted = st.form_submit_button(
        "Predict Brake Service Alert",
        use_container_width=True
    )


# --------------------------------
# 4. Send Data to FastAPI
# --------------------------------

if submitted:

    vehicle_data = {
        "vehicle_model": vehicle_model,
        "purchase_year": purchase_year,
        "vehicle_age_years": vehicle_age_years,
        "under_warranty": under_warranty,
        "branch_location": branch_location,
        "climate_region": climate_region,
        "financing_type": financing_type,
        "customer_type": customer_type,
        "self_reported_driving_type": self_reported_driving_type,
        "self_reported_terrain": self_reported_terrain,
        "number_of_service_visits_total": number_of_service_visits_total,
        "current_odometer_km": current_odometer_km,
        "km_since_last_service": km_since_last_service,
        "days_since_last_service": days_since_last_service,
        "pad_material": pad_material,
        "km_on_current_pads": km_on_current_pads,
        "pad_thickness_mm": pad_thickness_mm,
        "rotor_condition_score": rotor_condition_score,
        "brake_fluid_moisture_pct": brake_fluid_moisture_pct,
        "last_service_type": last_service_type,
        "last_service_cost_usd": last_service_cost_usd,
        "warranty_covered": warranty_covered,
        "avg_annual_service_spend_usd": avg_annual_service_spend_usd,
        "preferred_contact_channel": preferred_contact_channel,
        "previous_outreach_sent": previous_outreach_sent,
        "previous_outreach_response": previous_outreach_response
    }

    try:

        with st.spinner("Predicting brake service alert..."):

            response = requests.post(
                API_URL,
                json=vehicle_data,
                timeout=30
            )

        if response.status_code == 200:

            result = response.json()

            st.subheader("Prediction Result")

            if result["prediction"] == 1:
                st.error("⚠️ Service Alert Required")
            else:
                st.success("✅ No Service Alert Required")

            st.metric(
                "Alert Probability",
                f"{result['alert_probability'] * 100:.2f}%"
            )

            st.write(
                "Prediction:",
                result["prediction"]
            )

        else:
            st.error(
                f"API Error: {response.status_code}"
            )
            st.code(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. "
            "Make sure the backend is running."
        )

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
