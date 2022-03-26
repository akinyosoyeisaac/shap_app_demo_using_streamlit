import streamlit as st
import matplotlib.pyplot as plt
import shap
from model import *


def main():
    st.title("MODEL EXPLAINATION")
    st.subheader("the essence of the work is to try has much as possible to explain our logistic model")

    # subseting our dataset
    st.header("Slicing the Dataset")
    st.subheader("Selecting random observation from the dataset")
    start_index = st.number_input(label="Enter a value for the start index", min_value=0, max_value=150, value=0,
                                  step=10, key='start')
    end_index = st.number_input(label="Enter a value for the end index", min_value=0, max_value=150, value=150,
                                  step=10, key='end')
    selected_observation = X.iloc[start_index:end_index,:]

    # targeted Features
    feature = st.selectbox(label="select feature of interest", options=df.columns)

    # selecting the type of visualization
    list_visuals = ["partial dependence plots", "beeswarm", "scattered_plot"]
    select_visual = st.selectbox(label="Choose the your preferred model explainer from the list below", options=list_visuals, index=0)

    # Loading the shap value
    explainer = shap.Explainer(model_dt.predict, selected_observation)
    shap_values = explainer(X)

    # displaying our visuals
    st.markdown("# **VISUALIZATION OF SELECTED PLOT**")
    if select_visual == "partial dependence plots":
        sample_ind = 20
        fig, ax = plt.subplots()
        shap.plots.partial_dependence(feature, model_dt.predict, selected_observation, ice=False, model_expected_value=True,
                                      feature_expected_value=True, shap_values=shap_values[sample_ind:sample_ind+1,:], ax=ax)
        st.pyplot(fig)

    if select_visual == "scattered_plot":
        fig, ax = plt.subplots()
#         shap.plots.beeswarm(shap_values[:, feature], max_display=4)
        shap.plots.scatter(shap_values[:,feature], color=shap_values, ax=ax)
        st.pyplot(fig)
        
    if select_visual == "beeswarm":
        fig = plt.figure()
        shap.plots.beeswarm(shap_values[:, [feature]], max_display=4)
        st.pyplot(fig)
       

if __name__ == "__main__":
    model_dt.fit(X, y)
    main()




