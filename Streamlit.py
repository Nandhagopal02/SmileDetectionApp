import streamlit as st
import numpy as np
import pickle
from PIL import Image
from sklearn.preprocessing import StandardScaler


with open(r"D:\Smile Project Streamlit\SmileImageProject\model.pkl", "rb") as f:
    model = pickle.load(f)


with open(r"D:\Smile Project Streamlit\SmileImageProject\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


st.title("Smile Detection App 😊")
st.write("Upload an image and let the model predict if you're smiling!")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    
    st.image(uploaded_file, caption="Original Image", use_container_width=True)

    
    image = Image.open(uploaded_file).convert("L").resize((64, 64), Image.LANCZOS)

    image_array = np.array(image).flatten().reshape(1, -1)

    
    image_scaled = scaler.transform(image_array)

    
    prediction = model.predict(image_scaled)[0]  
    probability = model.predict_proba(image_scaled)[0]  

    
    smile_probability = probability[1] * 100 
    st.write(f"**Predicted Class:** {'😊 Smile' if prediction == 1 else '😐 No Smile'}")
    st.write(f"**Smile Probability:** {smile_probability:.2f}%")
