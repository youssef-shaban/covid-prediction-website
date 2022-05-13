import joblib
from skimage import feature
import cv2
clf= joblib.load("covid-prediction-website/model.joblib")

def predict(img):
    img = cv2.resize(img, (244, 244))
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_desc = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    pred= clf.predict(hog_desc.reshape(1,-1))
    return pred
