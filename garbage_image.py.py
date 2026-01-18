import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.resnet50(pretrained=False)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, xb):
        return self.network(xb)

@st.cache_resource
def load_assets():
    with open('classes.json', 'r') as f:
        class_names = json.load(f)
    
    
    model = ResNet(num_classes=len(class_names))
    
    checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names


try:
    model, class_names = load_assets()
except FileNotFoundError:
    st.error("Missing 'best_model.pth' or 'classes.json'. Please upload them to the app folder.")
    st.stop()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.sidebar.title("Garbage Image Classficiation Navigation")
pages = st.sidebar.selectbox("Choose the options:" , ['Introduction' , 'Imgae Classes and indentificaiton', 'About the Creator'])

if pages == 'Introduction':
    
    st.title("GARBAGE IMAGE CLASSIFICATION USING DEEP LEARNING")
    st.header("Deep Learning")
    st.write("I have built an AI-powered solution to automate waste segregation and improve recycling efficiency. By training a deep learning model on various waste categories, I achieved a high-accuracy classifier capable of distinguishing between recyclables and organic matter. I successfully deployed this system as an interactive user interface, demonstrating how computer vision can be used to solve real-world environmental challenges.")
    st.header("End - End Machine Leanring Pipeline")
    st.markdown("""
    * Python Scripting
    * Deep Learning
    * ImagePreprocessing & Augment
    * Evaluation Metrics
    * Streamlit App Development 
                """)
elif pages == 'Imgae Classes and indentificaiton':
    st.title("Imgar Classes")
    st.write("There are about 12 different types of trash classes")
    st.markdown("""
    * Battery 
    * Biological
    * Brown-glass
    * Cardboard
    * Clothes
    * Green-glass
    * Metal
    * Paper
    * Plastic
    * Shoes
    * Trash
    * White-glass
                """)
    
    st.header("Identification")
    st.write("In this identificaiton the user can upload the photo and can identfiy the class ")
    st.write("Upload an image, and the ResNet-50 model (93% accuracy) will identify it.")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Predict button
        if st.button('Identify Image'):
            with st.spinner('Analyzing...'):
                # Preprocess
                img_tensor = preprocess(image).unsqueeze(0)
                
                # Forward pass
                with torch.no_grad():
                    output = model(img_tensor)
                    # Convert output to probabilities
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    confidence, index = torch.max(probabilities, dim=0)
                
                # Show Results
                label = class_names[index]
                percent = confidence.item() * 100
                
                st.success(f"**Prediction:** {label}")
                st.info(f"**Confidence:** {percent:.2f}%")
                
                # Optional: Progress bar for confidence
                st.progress(confidence.item())
    
elif pages == 'About the Creator':
    st.title ("About the Creator")
    st.write("Gowthaman C")
    st.write("Data Scientic")
    st.write("gowthamcartigayane@gmail.com")
    