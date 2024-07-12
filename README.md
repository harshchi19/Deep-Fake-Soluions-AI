# VeriGuard: Deep Learning for Fake Detection
![47e42b7d329a80d76be3855c2a9151874a7f3f1e6c9f81645f79c66c](https://github.com/user-attachments/assets/fef850a5-6ca2-4917-b77c-ee28fdb8f8d9)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone Repository](#clone-repository)
  - [Create Virtual Environment](#create-virtual-environment)
  - [Activate Virtual Environment](#activate-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Run Application](#run-application)
- [Usage](#usage)
- [Model Information](#model-information)
  - [Image Fake Detection](#image-fake-detection)
  - [Audio Fake Detection](#audio-fake-detection)
  - [Account Fake Detection](#account-fake-detection)
  - [Fake News Detection](#fake-news-detection)
- [License](#license)
- [Contact](#contact)

## Overview
VeriGuard is a cutting-edge web application designed to classify various types of fake content including images, audio, accounts, and news. Utilizing advanced deep learning techniques, VeriGuard helps users identify fake content with ease and reliability.
![image](https://github.com/user-attachments/assets/52b1e868-6595-4609-b46a-8e77e5c36a0e)
![image](https://github.com/user-attachments/assets/7a54fe54-ad19-4d6b-82d4-bd6b5aab2075)

## Features
- **Image Fake Detection**: Upload an image to determine its authenticity.
- **Account Fake Detection**: Check the authenticity of account details.
- **Audio Fake Detection**: Upload an audio file to detect manipulations.
- **Fake News Detection**: Input text to verify its authenticity.
- **Detailed Model Information**: Insights into the models used for detection.

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)
- git (for cloning the repository)

### Clone Repository
1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/harshchi19/Deep-Fake-Soluions-AI.git
    ```

### Create Virtual Environment
2. Create a virtual environment to manage dependencies:
    ```bash
    python -m venv venv
    ```

### Activate Virtual Environment
3. Activate the virtual environment:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

### Install Dependencies
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Run Application
5. Run the application:
    ```bash
    streamlit run app.py
    ```

## Usage
1. Open the web application in your browser. The default address is `http://localhost:8501`.
2. Use the sidebar to select the task you want to perform (Image, Account, Audio, or News Fake Detection).
3. Follow the instructions provided for each task to upload files or input data.
4. Click on the 'Predict' or 'Check' button to get the results.

## Model Information

### Image Fake Detection
![image](https://github.com/user-attachments/assets/4b3500b6-41db-4cc7-9915-c17a3352e5d8)

- **Dataset**: The dataset was taken from Kaggle and includes 1288 faces, with 589 real and 700 fake images. You can find it [here](https://www.kaggle.com/hamzaboulahia/hardfakevsrealfaces).
- **Model Architecture**: 
  - Sequential model with 5 Convolutional Layers and 4 Dense Layers.
  - Max Pooling Layers introduced to reduce spatial dimensions.
  - ReLU activation in all layers except the output layer.
  - Softmax activation in the output layer.
- **Performance**: Achieved perfect classification on test samples.
- **Visualization**: 
  - Real vs Fake distribution bar chart.
  - Model architecture diagram.

### Audio Fake Detection
![image](https://github.com/user-attachments/assets/f4c71248-caf2-49dd-be60-3d21ee21d580)

- **Dataset**: Custom collected dataset with 10 audio files (7 real, 3 fake).
- **Model Architecture**: 
  - Sequential model with convolutional layers tailored for audio features.
  - Dense Layer with softmax activation for multi-class classification.
- **Performance**: Details on training and validation performance.

### Account Fake Detection
![image](https://github.com/user-attachments/assets/f1eabafa-36f3-4d5e-b597-6f855a374b87)

- **Model**: Analyzes various features of an account to determine its authenticity.
- **Features Considered**: Account age, activity patterns, network connections, etc.
- **Performance**: Details on the model’s accuracy and robustness.

### Fake News Detection
![image](https://github.com/user-attachments/assets/c64e1e1b-d5c9-4e02-864d-a0e81da5fcd7)
![image](https://github.com/user-attachments/assets/f144ce1e-2fc1-4f53-8110-ae395326b5df)

- **Model**: Text analysis to detect fake or manipulated news.
- **Techniques Used**: NLP techniques such as tokenization, sentiment analysis, and semantic analysis.
- **Performance**: Model accuracy, precision, recall, and F1-score.

### Code of Conduct
Please adhere to the [Code of Conduct](CODE_OF_CONDUCT.md) when contributing to this project.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
![image](https://github.com/user-attachments/assets/2027d427-e4c2-4cc2-9757-c9d105467c9f)

### Developers behind this project:
- **Arya Vaidya** (UI/UX Developer)
  - Email: aryavaidya59@gmail.com

- **Harsh Chitaliya** (Data Scientist)
  - Email: harshchitaliya010@gmail.com

- **Gaurav Khati** (Software Engineer)
  - Email: khatigaurav8@gmail.com

- **Morvi Panchal** (Machine Learning Engineer)
  - Email: morvieee5@gmail.com

## Live Demo : https://veriguard-ai.streamlit.app/
