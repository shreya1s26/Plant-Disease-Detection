import cv2
from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np
import streamlit as st
def imgch(img):
    lab_img=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab_img)
    equ=cv2.equalizeHist(l)
    updated_lab_img =cv2.merge((equ,a,b))
    return cv2.cvtColor(updated_lab_img,cv2.COLOR_LAB2BGR)

def prediction(model,img,size):
    test_img=imgch(img)
    test_img=cv2.resize(img,(size,size))
    test_input=test_img.reshape(1,size,size,3)
    return model.predict(test_input)[0,0]

def main():
    st.title('Plant Disease Detection')
   
    #st.markdown(
    #"""
    #<link rel="stylesheet" type="text/css" href="Background.jpg">
    #""",
    #unsafe_allow_html=True)
    img=st.file_uploader("Upload an Image......",type=["jpg",".webp","jpeg","png"])
    modelVGG16=load_model("Model_VGG16.h5")
    modelVGG19=load_model("Model_VGG19.h5")
    modelAN=load_model("AlexNetModel.h5")
    if img is not None:
        image=Image.open(img)
        img = np.asarray(image)
        col1,col2=st.columns(2)
        with col1:
            #image=image.resize((150,150))
            st.image(image)
        with col2:
            if st.button('Classify'):
                p2=prediction(modelVGG16,img,150)
                p3=prediction(modelVGG19,img,150)
                p1=prediction(modelAN,img,227)
                #z=p1+p2+p3/3
                if ((2*p1+3*p2+4*p3)/9)>0.50:
                    st.success("Healthy")
                    st.success(((2*p1+3*p2+4*p3)/9)*100)
                else:
                    st.success("Defected")
                    st.success((1-(2*p1+3*p2+4*p3)/9)*100)


if __name__=='__main__':
    main()
