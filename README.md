📘 Fruit Predictor

A Machine Learning-based Fruit Image Classification Web App built with Python, TensorFlow/Keras, and Flask. The app lets users upload fruit images and receive predicted fruit labels using a trained MobileNet model.

🧠 Project Overview

This project predicts the type of fruit from a user-uploaded image using a deep learning model trained with MobileNet architecture. The model classifies fruit images into different categories with high accuracy and provides an interactive web user interface for predictions.

📁 Project Structure
Fruit_Predictor/
├── app.py                       # Flask web app
├── fruit_mobilenet_model.h5     # Trained MobileNet model
├── class_indices.pkl            # Label mapping file
├── templates/                  # HTML templates directory
│   └── index.html
├── download.jpg                 # Example image
└── README.md                   # Project documentation
🛠️ Features

✔️ Upload fruit images from your device
✔️ Predict the fruit type using a trained model
✔️ Interactive web interface
✔️ Lightweight and easy to run locally

📌 Technologies Used

Python

Flask – Web framework

TensorFlow / Keras – Deep Learning

MobileNet – Pretrained model for image classification

HTML/CSS – Front-end UI

🧪 How to Run Locally
1. Clone the Repository
git clone https://github.com/vishnudilipmali/Fruit_Predictor.git
cd Fruit_Predictor
2. Create & Activate Virtual Environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
3. Install Dependencies
pip install -r requirements.txt

If you don’t have requirements.txt, install Flask and TensorFlow manually:

pip install flask tensorflow pillow numpy
4. Run the Flask App
python app.py

Open your browser and go to:

http://127.0.0.1:5000

You’ll see the upload interface where you can test fruit images 📸

📷 Example Usage

Upload an image (e.g., an apple photo) —
the app will display the predicted fruit name on the results page.

🧠 How It Works (Model)

The model used is a MobileNet-based CNN classifier trained to recognize fruit images. It takes input images, preprocesses them, and outputs class probabilities. The highest probability wins — that’s the predicted fruit label.

You can retrain or improve the model using your own dataset in future versions.

🤝 Contributing

Want to improve the project?

Add more fruit classes

Improve UI/UX

Add real-time webcam support

Deploy on cloud (Heroku / Railway / Vercel)

Feel free to open an issue or submit a pull request!
<img width="1906" height="1079" alt="image" src="https://github.com/user-attachments/assets/26ea59a2-330c-4754-a4c1-a4444ba2dde9" />
output:
<img width="1911" height="1079" alt="image" src="https://github.com/user-attachments/assets/72da6a33-5ce5-48d8-a119-de3d033d896f" />

