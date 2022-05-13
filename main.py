import streamlit as st
from PIL import Image
import numpy as np
import model
from streamlit_option_menu import option_menu

selected2 = option_menu(None, ["Home", "Behind Code", "About Us"], 
icons=['house', 'bi-body-text', "bi-bookmark"], 
menu_icon="cast", default_index=0, orientation="horizontal")


if selected2=="Home":
    st.write("""
            # X-ray Image Recognition
            ### our website can used to check if you have COVID-19 just from checking you X-ray scan on chest.
            so please don't be shy and upload your Scan
    """)
    uploaded_image = st.file_uploader("Upload you X-ray Scan", type= ['png','jpg','jpeg'])
    if uploaded_image is not None:
        img= Image.open(uploaded_image)
        
        st.image([img,model.hog_img(np.array(img))],width=250,clamp=True)
        prediction= model.predict(np.array(img))
        if(prediction==1):
            st.write("COVID")
        else:
            st.write("NO COVID")

elif selected2=="Behind Code":
    st.write("""
        # Code area
        ## this place is under construction
    """)

elif selected2=="About Us":
    st.write("""
    # About Us
    ## this place is under construction
    """)
