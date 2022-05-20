from turtle import right
import requests
import streamlit as st
from PIL import Image
import numpy as np
import model
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

st.set_page_config(page_title="COVID Predictions",layout="wide")

def load_lottieurl(url):
    r=requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_v6njxply.json")
tree_coding = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_lzhwcgzg.json")
img_agg = Image.open("images/Aggregiation.jpg")

# ---- HEADER SECTION ----
selected2 = option_menu(None, ["Home", "Explanation", "Build Random Forest"], 
icons=['house', 'bi-body-text', "bi-bookmark"], 
menu_icon="cast", default_index=0, orientation="horizontal")

#
if selected2=="Home":

    with st.container():
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("X-ray Image Recognition")
            st.write("""
                - Our website can used to check if you have COVID-19 just from checking you X-ray scan on chest.
                - Please upload your scan if interested!
            """)
            uploaded_image = st.file_uploader("", type= ['png','jpg','jpeg'])
            if uploaded_image is not None:
                img= Image.open(uploaded_image)
            
                st.image([img,model.hog_img(np.array(img))],width=250,clamp=True)
                prediction  = model.predict(np.array(img))
                if(prediction==1):
                    st.warning("We are sorry to say that, but you got COVID virus in your system")
                else:
                    st.success("The Predicisions says you are a normal person, Enjoy your life")
                    st.balloons()
        with right_column:
            st_lottie(lottie_coding,height=300, key="coding")

elif selected2=="Explanation":
    with st.container():
        left_column, right_column = st.columns((3,1))
        with left_column:
            st.header("Random forest classifier")
            st.write("""
                    Random forest is a supervised machine learning algorithm developed by Leo Breiman and Adele Cutler.

                    Supervised means that we have the inputs and the outputs, we are training the data.
                    """)
            st.subheader("How does it work?")
            st.write("""
                - It mixes the output of numerous decision trees to produce a single outcome which would be the output of the majority of trees. 
                - Its popularity is due to its ease of use and flexibility, since it can handle both classification and regression problems.
                - Classification: a problem whose outputs are categorical in nature (yes/no, true/false) > 1/0
                    """)

        with right_column:
            st_lottie(tree_coding,height=300, key="coding")

    with st.container():
        st.write("---")  
        st.header("What is decision tree?")   
        st.write("""
            It’s a binary tree that recursively splits the dataset until we are left with pure leaf nodes that represents the output.
                """)      

    with st.container():
        st.subheader("How does it work?")  
        st.write("""
           - Decision trees start with a basic question (go out for running) its answer is (yes or no), from there, you can ask a series of questions according to the features (weather, temperature, wind level) of the dataset to determine the answer
           - These questions make up the decision nodes in the tree, acting as a means to split the data. Each question helps an individual to arrive at a final decision, which would be denoted by the leaf node.
          
            Example of decision tree:
                """)
               
elif selected2=="Build Random Forest":
    with st.container():
        st.header("Bootstrapping")
        st.write("""

            - Build new datasets from the original data by selecting random rows with replacement and every dataset will contain the same number of rows as the original one. 
            - Train a decision tree on each of the bootstrapped datasets independently by using a subset of the features selected randomly for each tree. 
            - This method ensures creating a forest of ensemble of uncorrelated decision trees.

                """)
    with st.container():
        st.write("---")  
        st.header("Aggregation")
        st.write("""
 
                To make a prediction using this forest we take a new data point and pass it through each tree that will give a specific prediction and we will take the majority voting as a final result.

                    """)
        st.image(img_agg)

