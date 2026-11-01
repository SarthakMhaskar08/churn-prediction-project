import streamlit as st
import pandas as pd
import joblib

model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')
columns = joblib.load('models/columns.pkl')

st.title('Customer Churn Prediction')
st.write('Fill in the customer details below and the app will tell how likely the customer is to leave.')

st.subheader('Customer Details')

gender = st.selectbox('Gender', ['Female', 'Male'])
senior = st.selectbox('Senior Citizen', ['No', 'Yes'])
partner = st.selectbox('Has Partner', ['No', 'Yes'])
dependents = st.selectbox('Has Dependents', ['No', 'Yes'])

st.subheader('Account Details')

tenure = st.slider('Tenure (months)', 1, 72, 12)
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless = st.selectbox('Paperless Billing', ['No', 'Yes'])
payment = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly = st.number_input('Monthly Charges ($)', min_value=15.0, max_value=125.0, value=70.0)
total = st.number_input('Total Charges ($)', min_value=0.0, max_value=9000.0, value=800.0)


st.subheader('Services')

phone = st.selectbox('Phone Service', ['No', 'Yes'])
multiple = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
security = st.selectbox('Online Security', ['No internet service', 'No', 'Yes'])
backup = st.selectbox('Online Backup', ['No internet service', 'No', 'Yes'])
protection = st.selectbox('Device Protection', ['No internet service', 'No', 'Yes'])
support = st.selectbox('Tech Support', ['No internet service', 'No', 'Yes'])
tv = st.selectbox('Streaming TV', ['No internet service', 'No', 'Yes'])
movies = st.selectbox('Streaming Movies', ['No internet service', 'No', 'Yes'])


if st.button('Predict Churn'):

    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior == 'Yes' else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone],
        'MultipleLines': [multiple],
        'InternetService': [internet],
        'OnlineSecurity': [security],
        'OnlineBackup': [backup],
        'DeviceProtection': [protection],
        'TechSupport': [support],
        'StreamingTV': [tv],
        'StreamingMovies': [movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless],
        'PaymentMethod': [payment],
        'MonthlyCharges': [monthly],
        'TotalCharges': [total]
    })

    input_encoded = pd.get_dummies(input_data, dtype=int)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

    probability = model.predict_proba(input_encoded)[0][1]

    st.subheader('Result')
    st.write(f'Churn Probability: {probability * 100:.1f}%')

    if probability >= 0.5:
        st.error('High risk: this customer is likely to leave.')
    else:
        st.success('Low risk: this customer is likely to stay.')   