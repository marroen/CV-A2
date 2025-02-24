
import cv2 as cv
import numpy as np

from background_subtraction import compute_gaussian_model
from background_subtraction import background_subtraction


# Background and foregrond videos
background_vid = cv.VideoCapture("data/cam1/background.avi")
foreground_vid = cv.VideoCapture("data/cam1/video.avi")

# Video resolution
width = 644
height = 486
resolution = (width, height)

# Compute the background model for cam1
cam1_background_model = compute_gaussian_model(resolution, vid=background_vid)

# Use the cam1 background model for background subtraction on the foreground video
background_subtraction(cam1_background_model, foreground_vid, resolution)