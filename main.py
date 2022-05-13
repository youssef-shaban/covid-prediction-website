import streamlit as st
from PIL import Image
import cv2
import numpy as np
import model
st.write("""
        # X-ray Image Recognition
        ### our website can used to check if you have COVID-19 just from checking you X-ray scan on chest.
        so please don't be shy and upload your Scan
""")
uploaded_image = st.file_uploader("Upload you X-ray Scan", type= ['png','jpg','jpeg'])
if uploaded_image is not None:
    img= Image.open(uploaded_image)
    
    st.image([img,img],width=250)
    prediction= model.predict(np.array(img))
    if(prediction==1):
        st.write("COVID")
    else:
        st.write("NO COVID")
