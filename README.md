# Sign Language Recognition Model  
This project is a sign language recognition model that uses computer vision techniques to recognize sign language gestures in real-time.

## Overview  
The model is built using the MediaPipe framework for hand detection and tracking, and a custom deep learning model for gesture recognition. It uses a dataset of sign language gestures to train the model, and achieves an accuracy of 86% on the test set.

## Requirements  
* Python 3.6+  
* OpenCV  
* MediaPipe  
* TensorFlow  
* NumPy  

## Installation  
Clone the repository:  
```git clone https://github.com/mithz-z/Sign-Language-Recognition-Using-LSTM-in-Streamlit.git```

## Install the required packages:  
```pip install -r requirements.txt```

## Run the application (must run inside an IDE):  
```streamlit run Streamlit.py```

## Usage  
1. Run the Streamlit.py file to start the application.
2. Use the webcam to capture your hand gestures.
3. The model will recognize the gestures in real-time and display the result on the screen.

## License  
This project is licensed under the MIT License - see the LICENSE file for details.
