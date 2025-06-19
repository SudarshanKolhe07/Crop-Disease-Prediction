# 🌾 AgriSense AI

AgriSense AI is a smart and user-friendly web-based platform designed to assist farmers by offering early crop disease detection and personalized crop recommendations. By combining **Convolutional Neural Networks (CNNs)**, weather data, and historical agricultural trends, this system empowers farmers to make timely, informed decisions for healthy and sustainable crop production.

## 🚀 Features

- 🔍 **Plant Disease Detection** using CNNs with 90%+ accuracy
- 🌦️ **Crop Recommendation** based on weather conditions and soil data
- 📸 **Image Upload Interface** for real-time disease identification
- 🌐 **Multilingual UI** for ease of access to non-technical users
- 📊 **Historical Data Utilization** to improve prediction performance
- 📱 **Responsive Web Design** compatible with mobile and desktop

## 💡 Tech Stack

- **Frontend**: HTML, CSS, Bootstrap, JavaScript
- **Backend**: Python (Flask), TensorFlow/Keras (CNN Models)
- **Database**: MongoDB / SQL (Configurable)
- **Tools & APIs**: Git, GitHub, REST APIs, Figma (UI Design)

## 🧠 How It Works

1. Users upload a crop image via the web interface.
2. The CNN model classifies the disease based on visual features.
3. The system matches the detected issue with curated treatment guidelines.
4. Based on weather and soil data, suitable crops are also recommended.

## 📁 Folder Structure

AgriSense-AI/
├── model/ # Trained CNN models
├── static/ # CSS, JS, images
├── templates/ # HTML templates
├── app.py # Main Flask application
├── dataset/ # Sample images for testing
├── requirements.txt # Python dependencies
└── README.md
