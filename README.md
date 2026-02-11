# OCT Retinal Disease Classification - Streamlit Deployment

This project demonstrates the deployment of a Convolutional Neural Network (CNN) trained on OCT (Optical Coherence Tomography) retinal images to classify retinal diseases. The model is served using a Streamlit web application, allowing users to upload OCT images and receive real-time predictions with confidence scores.

---

##  Demo

 Live App: https://oct-image-classifier.streamlit.app/

---

## Model Overview

- Architecture: Custom CNN with
  - Convolutional blocks
  - Batch Normalization
  - ReLU activations
  - MaxPooling
  - Dropout for regularization
- Input: Grayscale OCT images (28×28)
- Output Classes:
  - CNV (Choroidal Neovascularization)
  - DME (Diabetic Macular Edema)
  - DRUSEN
  - NORMAL

---

## How to Run Locally

### Clone the Repository
git clone https://github.com/your-username/oct-streamlit-app.git
cd oct-streamlit-app

### Create Environment & Install Dependencies
pip install -r requirements.txt

### Run the Streamlit App
streamlit run app.py