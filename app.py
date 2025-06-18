import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
onehot_encoders = joblib.load("onehot_encoders.pkl")

with open("training_columns.txt", "r") as f:
    training_columns = f.read().split(",")

st.title("Sales Forecasting System")
st.subheader("Please enter the following data:")


order_date_str = st.text_input("Order date(Order Date)", value="2023-01-01")  # يمكن للمستخدم إدخال التاريخ بأي صيغة
segment = st.selectbox(" (Segment)", ["Consumer", "Corporate", "Home Office"])
promotion_flag_str = st.selectbox("Is there a promotion?(PromotionFlag)", ["Yes", "No"])
promotion_flag = 1 if promotion_flag_str == "Yes" else 0
product_id = st.text_input(" (Product ID)")
country = st.text_input(" (Country)")

if st.button("Sales forecast"):
    try:
        order_date = pd.to_datetime(order_date_str, errors='coerce')

        if pd.isna(order_date):
            st.error("Invalid date! Please enter a valid date.")
        else:
            input_data = pd.DataFrame([{
                "Order Date": order_date,
                "Segment": segment,
                "PromotionFlag": int(promotion_flag),
                "Product ID": product_id,
                "Country": country
            }])

            input_data["Year"] = input_data["Order Date"].dt.year
            input_data["Month"] = input_data["Order Date"].dt.month
            input_data["Weekday"] = input_data["Order Date"].dt.weekday
            input_data["IsWeekend"] = input_data["Weekday"].isin([5, 6])

            input_data["Country"] = encoders["Country"].transform(input_data["Country"])
            input_data["Product ID"] = encoders["Product"].transform(input_data["Product ID"])

            segment_encoded = onehot_encoders["Segment"].transform(input_data[["Segment"]]).toarray()
            segment_df = pd.DataFrame(
                segment_encoded,
                columns=onehot_encoders["Segment"].get_feature_names_out(["Segment"]),
                index=input_data.index
            )

            processed = pd.concat([
                input_data[["Year", "Month", "Weekday", "IsWeekend", "PromotionFlag", "Country", "Product ID"]],
                segment_df
            ], axis=1)

            processed = processed.reindex(columns=training_columns, fill_value=0)

            prediction = model.predict(processed)
            st.success(f"Expected sales value: {prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"An error occurred while predicting: {e}")
