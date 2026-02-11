import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
class_names = {
    0: "CNV (Choroidal Neovascularization)",
    1: "DME (Diabetic Macular Edema)",
    2: "DRUSEN",
    3: "NORMAL"
}
class BestCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        self.fc1 = nn.Linear(128*3*3, 256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)

device = torch.device("cpu")
model = BestCNN()
model.load_state_dict(torch.load("best_model.pkl", map_location=device))
model.eval()


st.title("OCT Image Classification Demo")
st.write("Upload an OCT image and see the predicted class.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  
    st.image(image, caption='Uploaded OCT Image', use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(image).unsqueeze(0)  

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = class_names[torch.argmax(probs, dim=1).item()]
        confidence = probs[0, torch.argmax(probs, dim=1).item()].item()
    if confidence < 0.52:
        st.write("Low confidence prediction. The uploaded image may not be a valid OCT scan.")
    else:
        st.write(f"**Predicted Class:** {pred_class}")
        st.write(f"**Confidence:** {(confidence*100):.2f}%")

st.warning("Note: This model is trained only on retinal OCT images. Uploading non-OCT images (e.g., cats, people, objects) will produce unreliable predictions.")
