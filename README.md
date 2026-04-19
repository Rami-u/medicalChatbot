# MediBot — AI-powered Medical Symptom Chatbot

A simple, fast, and highly realistic web application built to classify medical symptoms and return predefined smart responses using Machine Learning (TF-IDF + SVM).

## 🚀 Features
- **Machine Learning**: Uses Term Frequency-Inverse Document Frequency (TF-IDF) and a Support Vector Machine (SVM) instead of Large Language Models.
- **Predefined Intents**: Automatically identifies medical intents like *Allergy, Cold, Flu, Anemia, Hypertension, Migraine, Pneumonia, etc.*
- **Clean Architecture**: A clear separation between training the model, serving inferences, and the Flask API.
- **Premium UI**: Features a beautiful dark-mode chat interface with glassmorphism, animated backgrounds, responsive sidebars, typing indicators, and a confidence probability bar.

## 📁 Project Structure

```
medicalChatbot/
├── train.py            # Script to parse intents.json and train the TF-IDF + SVM model.
├── model.py            # Exposes a predict() function to load the model and evaluate text.
├── responses.py        # Contains the mapping of intents to predefined smart responses.
├── chat.py             # Flask application to serve the backend API and frontend UI.
├── requirements.txt    # Python package dependencies.
├── intents.json        # Training dataset (1,800 examples mapped to 9 medical conditions).
├── models/             # Directory where .pkl model artifacts are saved.
└── static/             # Directory containing frontend assets.
    ├── index.html      # The premium UI layout.
    ├── style.css       # Clean, modern, dark-mode CSS styling.
    └── app.js          # Logic for chat interactions, API fetching, and animations.
```

## 🛠️ Step-by-step Execution Guide

### 1. Set Up Environment
It is recommended to use a virtual environment.
```bash
python -m venv .venv

# Activate it (Windows):
.\.venv\Scripts\activate

# Or activate it (macOS/Linux):
source .venv/bin/activate
```

### 2. Install Dependencies
Install the required packages such as Flask and Scikit-Learn:
```bash
pip install -r requirements.txt
```

### 3. Train the Model
Run the training script to read the examples from `intents.json`, convert the strings to features via TF-IDF, train the SVM classifier, and save the binary files into the `models/` folder.
```bash
python train.py
```
> *You should see a printout of the test accuracy (~100%).*

### 4. Start the Application
Run the Flask server, which will automatically load the saved `.pkl` model and serve both the API and the user interface.
```bash
python chat.py
```

### 5. Start Chatting!
Open your browser and visit:  
👉 **http://127.0.0.1:5000/**

You can type phrases like *"I have a severe headache and nausea"* or click on the quick example chips to see the bot predict the symptom, display the confidence score, and reply with the predefined medical response.