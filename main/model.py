import cv2
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import glob
import os

class BeamClassifier:
    def __init__(self):
        # Initialize the classifier
        self.model = SVC()

    def load_images(self, good_subpath, bad_subpath):
        # Load all images and label them
        
        # Get the directory where the script is located
        script_dir = os.path.dirname(__file__)
        
        # Construct the full paths to the 'good' and 'bad' directories
        good_path = os.path.join(script_dir, good_subpath)
        bad_path = os.path.join(script_dir, bad_subpath)
        
        good_images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).flatten() for file in glob.glob(os.path.join(good_path, '*.png'))]
        bad_images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).flatten() for file in glob.glob(os.path.join(bad_path, '*.png'))]

        # Combine and create labels
        X = good_images + bad_images
        y = ['normal distribution'] * len(good_images) + ['inconclusive'] * len(bad_images)
        return X, y

    def train(self, X, y):
        # Split the dataset into training and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = self.model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Model accuracy: {accuracy}')

        # Save the model
        joblib.dump(self.model, 'beam_classifier_model.pkl')

    def classify(self, image_path):
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).flatten()

        # Predict
        prediction = self.model.predict([image])
        return prediction[0]

# Usage
# Create a new classifier
#classifier = BeamClassifier()

# Load and label images
#X, y = classifier.load_images('./model_datasets/normal distribution', './model_datasets/inconclusive')

# Train the classifier
#classifier.train(X, y)

# Classify a new image
#classification = classifier.classify('Z:/VSCode/Project154-1/main/static/processed/2_Burst78_Shot8/2_Burst78_Shot8_final_processed.png')
#classification = classifier.classify('Z:/VSCode/Project154-1/main/static/processed/run04_Shot7/run04_Shot7_final_processed.png')
#print(f"The image was classified as {classification}")
