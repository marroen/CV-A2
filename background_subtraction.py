# Using this for testing, getting a head start on the background subtraction

import cv2 as cv
import numpy as np

# Open the background video
vid = cv.VideoCapture("data/cam1/background.avi")

# Video resolution
width = 644
height = 486
resolution = (width, height)

# Create separate Gaussian models for each camera
# Stores the mean and variance of HSV values for each pixel in the background
# Eg. for pixel (y,x): [y, x, mean_H, mean_S, mean_V, var_H, var_S, var_V]
cam1_background_model = np.zeros((height, width, 6), dtype=np.float32)
cam2_background_model = np.zeros((height, width, 6), dtype=np.float32)
cam3_background_model = np.zeros((height, width, 6), dtype=np.float32)
cam4_background_model = np.zeros((height, width, 6), dtype=np.float32)

# Set variance for H, S, V to 1 (to avoid dividing by 0)
for model in [cam1_background_model, cam2_background_model, cam3_background_model, cam4_background_model]:
    model[:, :, 3:] = 1

def computeBackgroundModel(background_model, vid, resolution):

    width, height = resolution
    hsv_frames = []

    # Iterate through each frame
    frame_count = 0
    while True:

        success, frame = vid.read()  # Reads the next frame
        if not success:
            cv.imshow("Frame", last_frame)
            key = cv.waitKey(0)
            break  # Break when video ends

        # Convert frame to HSV
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Append frame to array
        hsv_frames.append(hsv_frame.copy())

        print(f"Processing frame {frame_count}")

        frame_count += 1
        last_frame = frame

    # Convert frames to numpy array
    # Eg. [num_frames, height, width, H, S, V]
    hsv_frames = np.array(hsv_frames)

    # Iterates through each HSV value of each pixel of each frame, and calculates the mean and variance
    cam1_background_model[:, :, :3] = np.mean(hsv_frames, axis=0)
    cam1_background_model[:, :, 3:] = np.var(hsv_frames, axis=0)

    vid.release()
    print(f"\nProcessed {frame_count} frames")



computeBackgroundModel(cam1_background_model, vid, resolution)


# TEST to see if the background model worked

test_pixels = [(50, 100), (200, 300), (400, 500)]

# Print mean and variance for test pixels
for y, x in test_pixels:
    mean_H, mean_S, mean_V = cam1_background_model[y, x, :3]
    var_H, var_S, var_V = cam1_background_model[y, x, 3:]

    print(f"\nPixel ({y}, {x}):")
    print(f"  Mean HSV: ({mean_H:.2f}, {mean_S:.2f}, {mean_V:.2f})")
    print(f"  Variance HSV: ({var_H:.6f}, {var_S:.6f}, {var_V:.6f})")
    print("-" * 40)