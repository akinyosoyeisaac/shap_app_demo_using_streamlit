import streamlit as st
import matplotlib.pyplot as plt
import shap
from model import *


def main():
    st.title("MODEL EXPLAINATION")
    st.subheader("the essence of the work is try has much as possible to explain our decision try model")

    # subseting our dataset
    st.header("Slicing the Dataset")
    st.subheader("Selecting random observation from the dataset")
    start_index = st.number_input(label="Enter a value for the first index", min_value=1, max_value=150, value=1,
                                  step=10, key='start')
    end_index = st.number_input(label="Enter a value for the first index", min_value=1, max_value=150, value=150,
                                  step=10, key='end')
    selected_observation = X.iloc[start_index:end_index,:]

    # targeted Features
    feature = st.selectbox(label="select feature of interest", options=df.columns)

    # selecting the type of visualization
    list_visuals = ["partial dependence plots", "beeswarm"]
    select_visual = st.selectbox(label="Choose the your preferred model explainer from the list below", options=list_visuals, index=0)

    # Loading the shap value
    explainer = shap.Explainer(model_dt.predict, selected_observation)
    shap_values = explainer(X)

    # displaying our visuals
    st.markdown("# **VISUALIZATION OF SELECTED PLOT**")
    if select_visual == "partial dependence plots":
        fig = plt.figure(figsize=(5,5))
        shap.plots.partial_dependence(feature, model_dt.predict, selected_observation, ice=False, model_expected_value=True,
                                      feature_expected_value=True)
        st.pyplot(fig=fig)

    if select_visual == "beeswarm":
        fig = plt.figure(figsize=(5,5))
        shap.plots.beeswarm(shap_values, max_display=4)
        st.pyplot(fig=fig)

if __name__ == "__main__":
    model_dt.fit(X, y)
    main()




