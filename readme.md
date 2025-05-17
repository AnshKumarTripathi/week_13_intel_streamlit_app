Screw Anomaly Detector
An AI-powered application for quality control and anomaly detection in screw manufacturing.
Show Image
Overview
This Streamlit application uses machine learning to detect anomalies in screw images. It can analyze images uploaded by users, captured through a camera, or selected from sample images to determine if a screw is normal or has defects.
The app is built with Streamlit and TensorFlow, making it easy to deploy and use in manufacturing environments for automated quality control.
Features

Anomaly Detection: Automatically identifies defects in screws
Multiple Input Methods:

File upload for existing images
Camera input for real-time inspection
Sample image selection for testing

Visual Results: Clear success/error indicators and confidence scores
User-Friendly Interface: Simple, intuitive design for all users

Installation
Prerequisites

Python 3.8 or higher
pip (Python package installer)

Setup

Clone this repository:
cd screw-anomaly-detector

Install dependencies:
bashpip install -r requirements.txt

Ensure your TensorFlow model files are in the project directory:

saved_model.pb
variables/ folder

Usage

Start the application:
bashstreamlit run app.py

Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501)
Select an input method:

File Uploader: Upload an image file from your computer
Camera Input: Use your webcam to capture an image
Sample Images: Select from pre-loaded sample images

Click the "Analyse Screw" button to process the image
Review the results showing:

Classification (Normal or Anomaly)
Confidence score
Inspection status with visual indicators

Model Information
The application uses a TensorFlow model trained to classify screw images as either "Normal" or "Anomaly". The model expects images to be resized to 224x224 pixels and normalized to values between 0 and 1.
Sample Images
The assets folder contains sample images for testing:

MF_000.png
MH_000.png
MN_000.png
ThS_000.png
ThT_000.png

Directory Structure
screw-anomaly-detector/
├── app.py # Main application file
├── assets/ # Sample images
│ ├── MF_000.png
│ ├── MH_000.png
│ ├── MN_000.png
│ ├── ThS_000.png
│ └── ThT_000.png
├── saved_model.pb # TensorFlow model file
├── variables/ # Model variables
├── README.md # Documentation
└── requirements.txt # Dependencies
Customization
Changing Model Parameters
To use a different model or adjust parameters:

Update the MODEL_PATH variable in app.py if your model is in a different location
Modify IMAGE_WIDTH and IMAGE_HEIGHT to match your model's expected input size
Adjust CLASS_NAMES if your model uses different class labels

Adding More Input Methods
New input methods can be added by:

Extending the input_method radio button options
Adding a new condition in the input method selection code
Including the new image source in the analysis section

Troubleshooting
Common Issues

Model loading error: Ensure the model files are in the correct location and format
Image processing error: Check that the image format is supported (jpg, jpeg, png)
Camera not working: Verify browser permissions for camera access

Error Messages

"Error loading AI model": Check model path and file integrity
"Error during prediction": Ensure image format is compatible with the model

Development
Contributing

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

Future Enhancements

Batch processing for multiple images
Historical tracking of inspection results
Integration with manufacturing systems
Mobile-optimized interface

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgements

TensorFlow for the machine learning framework
Streamlit for the web application framework
Contributors and testers who helped improve the application
