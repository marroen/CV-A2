
import cv2 as cv
import numpy as np

# Computes the gaussian distribution given EITHER a video OR an hsv_frames array and returns the gaussian model
def compute_gaussian_model(resolution, vid=None, hsv_frames=None):

    width, height = resolution

    # Stores the mean and variance of HSV values for each pixel in the frame
    # Eg. for pixel (y,x): [y, x, [mean_H, mean_S, mean_V, var_H, var_S, var_V]]
    gaussian_model = np.zeros((height, width, 6), dtype=np.float32)

    # Initialize variance for H, S, V to small value (to avoid dividing by 0 later)
    gaussian_model[:, :, 3:] = 1e-6

    # If passed a video, process video
    if hsv_frames is None:
        hsv_frames = []
        frame_count = 0

        # Iterate through each frame
        while True:

            success, frame = vid.read()  # Reads the next frame
            if not success:
                break  # Break when video ends

            hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV) # Convert frame to HSV
            hsv_frames.append(hsv_frame.copy()) # Append frame to array

            #print(f"compute_gaussian_model: Processing frame {frame_count}")
            frame_count += 1
        
        vid.release()
        print(f"compute_gaussian_model: Processed {frame_count} frames\n")

    # Convert frames to numpy array
    # Eg. [num_frames, height, width, [H, S, V]]
    hsv_frames = np.array(hsv_frames)

    # Iterates through each HSV value of each pixel of each frame, and calculates the mean and variance
    # TODO: find other algs for variation https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    gaussian_model[:, :, :3] = np.mean(hsv_frames, axis=0)
    gaussian_model[:, :, 3:] = np.var(hsv_frames, axis=0)

    return gaussian_model

# CHOICE TASK
# Automatically determines the distance threshold for background subtraction
# Brighter pixel = logarithmically larger threshold
def auto_threshold(background_model, multiplier=8):

    # Gets the intensity (V channel) of every pixel
    pixel_intensity = background_model[:, :, 2] + 1e-6

    # Brighter pixel = logarithmically larger threshold
    threshold = multiplier * np.log1p(pixel_intensity)

    return threshold # Returns threshold array containing custom threshold for every pixel

# Finds the largest object in a foreground frame and returns the frame with only that object
def extract_largest_object(foreground_frame):

    # Find all contours (object outlines)
    contours, _ = cv.findContours(foreground_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Find largest object in the scene
    largest_obj = max(contours, key=cv.contourArea)

    obj_frame = np.zeros_like(foreground_frame) # Create an empty frame

    # Fills the object contour with white pixels and inserts into obj_frame
    cv.drawContours(obj_frame, [largest_obj], -1, 255, thickness=cv.FILLED)

    return obj_frame

# Generate a colorised foreground frame with black background
def colorise_foreground(original_frame, foreground_frame):

    # Convert foreground frame to binary
    foreground_binary = cv.threshold(foreground_frame, 127, 255, cv.THRESH_BINARY)[1]
    
    # Get HSV representation of binary frame
    foreground_colored = cv.cvtColor(foreground_binary, cv.COLOR_GRAY2BGR)
    
    # Combine with the original frame
    colorised_foreground_frame = cv.bitwise_and(original_frame, foreground_colored)
    
    return colorised_foreground_frame

# Performes background subtraction on the given video compared to the given background model
def background_subtraction(background_model, vid, resolution):

    width, height = resolution
    foreground_silhouettes = []
    colorised_foreground_silhouettes = []

    # Iterate through each frame
    frame_count = 0
    while True:

        success, frame = vid.read()  # Reads the next frame
        if not success:
            break  # Break when video ends
        elif frame_count >= 100:
            break # Break when 100 frames are processed

        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV) #Convert frame to HSV

        # Difference of frame HSV and background mean HSV
        # Eg. (foreground mean HSV [05, 20, 50]) - (background mean HSV [50, 50, 05]) = [45, 30, 0]
        mean_diff = hsv_frame - background_model[:, :, :3]

        # Finds the distance between the foreground frame value and the background model distribution
        mahalanobis_dist = (
            (mean_diff[:, :, 0] ** 2) / (background_model[:, :, 0] + 1e-6) + # H
            (mean_diff[:, :, 1] ** 2) / (background_model[:, :, 1] + 1e-6) + # S
            (mean_diff[:, :, 2] ** 2) / (background_model[:, :, 2] + 1e-6)   # V
        )

        foreground_frame = np.zeros(mahalanobis_dist.shape, dtype=np.uint8) # Initializes foreground frame with all 0's

        threshold = auto_threshold(background_model)

        foreground_frame[mahalanobis_dist > threshold] = 255 # Set any pixel outide threshold to white
        foreground_frame = extract_largest_object(foreground_frame) # Only keeps the largest object

        colorised_foreground_frame = colorise_foreground(frame, foreground_frame) # Gets the colorised foreground 
        
        colorised_foreground_silhouettes.append(colorised_foreground_frame.copy())
        foreground_silhouettes.append(foreground_frame.copy())

        frame_count += 1

    vid.release()
    print(f"background_subtraction: Processed {frame_count} frames\n")
    return foreground_silhouettes, colorised_foreground_silhouettes

