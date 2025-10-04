import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and preprocessing objects
model = joblib.load('best_bagging_lr_model.pkl')
onehot_encoder = joblib.load('onehot_encoder.pkl')
standard_scaler = joblib.load('minmax_scaler.pkl')

st.title("Course Approval Prediction")
st.write("Enter the student's information to predict course approval.")

# Get user input for 'Felder' and 'Examen_admisión_Universidad'
felder_options = ['activo', 'visual', 'equilibrio', 'intuitivo', 'reflexivo', 'secuencial', 'sensorial', 'verbal']
selected_felder = st.selectbox("Select Felder category:", felder_options)
examen_admision = st.number_input("Enter Examen de Admisión score:", min_value=0.0, max_value=10.0, value=5.0)

# Create a DataFrame from user input
user_data = pd.DataFrame({'Felder': [selected_felder], 'Examen_admisión_Universidad': [examen_admision]})

# Preprocess the user input data
# Apply one-hot encoding to 'Felder'
felder_encoded_user = onehot_encoder.transform(user_data[['Felder']])
felder_encoded_user_df = pd.DataFrame(felder_encoded_user, columns=onehot_encoder.get_feature_names_out(['Felder']))

# Apply standard scaling to 'Examen_admisión_Universidad'
examen_admision_scaled_user = standard_scaler.transform(user_data[['Examen_admisión_Universidad']])
examen_admision_scaled_user_df = pd.DataFrame(examen_admision_scaled_user, columns=['Examen_admisión_Universidad_scaled'])

# Concatenate the preprocessed features
# Ensure all columns expected by the model are present, even if the user input doesn't cover all Felder categories
# Get the expected column names from the onehot_encoder
expected_felder_cols = onehot_encoder.get_feature_names_out(['Felder'])
# Create a DataFrame with all expected Felder columns, initialized to 0
all_felder_cols_df = pd.DataFrame(0, index=[0], columns=expected_felder_cols)
# Update the values for the selected Felder category
all_felder_cols_df[f'Felder_{selected_felder}'] = 1

# Concatenate the scaled numerical feature and the one-hot encoded Felder features
processed_user_data = pd.concat([examen_admision_scaled_user_df, all_felder_cols_df], axis=1)


# Make a prediction
if st.button("Predict Course Approval"):
    prediction = model.predict(processed_user_data)

    if prediction[0] == 1:
        st.success("Prediction: Approved")
    else:
        st.error("Prediction: Not Approved")
