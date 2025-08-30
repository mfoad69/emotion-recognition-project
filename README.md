# emotion-recognition-project

This project implements an automatic facial emotion recognition system using Python, OpenCV, and DeepFace. It detects faces in images and predicts emotions such as happy, sad, angry, fear, surprise, disgust, and neutral.

## Installation
1. Install Python 3.8 or higher.
2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   
The requirements.txt file includes:

opencv-python==4.5.5.64
deepface==0.0.79
numpy==1.23.5


Ensure the Haar Cascade file (haarcascade_frontalface_default.xml) is available in OpenCV's default path (usually in cv2.data.haarcascades). If not, download it from the OpenCV GitHub repository.

Usage

1. Place input images (JPG or PNG) in a folder (e.g., sample_images/).
2. Update the folder_path variable in emotion_recognition.py to point to your image folder:

folder_path = "path/to/your/image/folder"

3. Run the code:
python emotion_recognition.py

4. The script processes images in batch, detects faces using Haar Cascade, predicts emotions using DeepFace, and saves outputs with a green rectangle around faces and emotion labels (e.g., "Happy: 85.3%") with the prefix output_. Press Esc to skip image display or wait 2 seconds.

Repository Structure

emotion_recognition.py: Main script for emotion recognition.
requirements.txt: List of required Python libraries.
sample_images/: Folder containing sample images for testing.
README.md: This file, providing project overview and instructions.

Notes

Sample images in sample_images/ are provided for quick testing. You can replace them with your own images.
If no face is detected in an image (e.g., due to low light or unusual angles), an error message is printed, and the original image is saved unchanged.
For datasets like FER2013 or AffectNet, ensure images are in JPG/PNG format or extract them from .h5 files using libraries like h5py.

Example Output
Processed images will have a green rectangle around detected faces with an emotion label, e.g., "Happy: 85.3%".














