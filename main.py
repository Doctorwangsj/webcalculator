# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("A machine learning-based prediction model for in-hospital mortality among critically ill patients with hip fracture: an internal and external validated study")
st.sidebar.title("Parameters Selection")
st.sidebar.markdown("Choosing variables based on real conditions")

x1 = st.sidebar.selectbox("Gender", ("Male", "Female"))
x3 = st.sidebar.selectbox("Age", ("<65", "≥65 and <75", "≥75 and <85", "≥85"))
x4 = st.sidebar.selectbox("Anemia", ("No", "Yes"))
x9 = st.sidebar.selectbox("Mechanical ventilation", ("No", "Non-invasive", "Invasive"))
x24 = st.sidebar.selectbox("Cardiac arrest", ("No", "Yes"))
x26 = st.sidebar.selectbox("Chronic airway obstruction", ("No", "Yes"))

if st.button("Submit"):
    rf_clf = jl.load("Xgbc_clf_final_round.pkl")
    x = pd.DataFrame([[x1, x3, x4, x9, x24, x26]],
                     columns=["x1", "x3", "x4", "x9", "x24", "x26"])
    x = x.replace(["Male", "Female"], [0, 1])
    x = x.replace(["<65", "≥65 and <75", "≥75 and <85", "≥85"], [1, 2, 3, 4])
    x = x.replace(["No", "Yes"], [0, 1])
    x = x.replace(["No", "Non-invasive", "Invasive"], [1, 2, 3])
    x = x.replace(["No", "Yes"], [0, 1])
    x = x.replace(["No", "Yes"], [0, 1])
    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.text(f"Probability of in-hospital death: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.1275:
        st.text(f"Risk group: low-risk group")
    else:
        st.text(f"Risk group: High-risk group")

st.subheader('Model Introduction')
st.markdown('The web calculator was created based on the eXGBoosting machine. Internal validation: AUC value: 0.797. External validation: AUC value: 0.715; Accuracy: 0.788. Risk stratification: the cut-off value (12.75%) was defined as the average of the thresholds in the internal and external validation cohort. '
            'Patients with a predicted probability of less than 12.75% were classified into the low-risk group; patients with a predicted probability of 12.75% or above were classified into the high-risk group.')

st.subheader('How to use?')
st.markdown('Users could obtain the likelihood of in-hospital death for cases by selecting parameters that characterize the clinical condition of critically ill patients with hip fracture and then clicking ‘Submitting’ button.')