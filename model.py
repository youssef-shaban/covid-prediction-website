import joblib
from skimage import feature
import cv2
from pathlib import Path
clf= joblib.load(Path(__file__).parents[0]/"model.joblib")

def predict(img):
    img = cv2.resize(img, (244, 244))
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_desc = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    pred= clf.predict(hog_desc.reshape(1,-1))
    return pred

def hog_img(img):
    img = cv2.resize(img, (244, 244))
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, hog_image = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', visualize=True)
    hog_image = hog_image.astype('float64')
    return hog_image
    