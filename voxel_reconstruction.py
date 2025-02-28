
import cv2 as cv
import numpy as np

from background_subtraction import compute_gaussian_model
from background_subtraction import background_subtraction
from calibration_processing import load_config

# Stores each
cams = ["cam1", "cam2", "cam3", "cam4"]

# Store videos in dictionary for each camera
background_videos = {cam: cv.VideoCapture(f"data/{cam}/background.avi") for cam in cams}
foreground_videos = {cam: cv.VideoCapture(f"data/{cam}/video.avi") for cam in cams} 

# Video resolution
width = 644
height = 486
resolution = (width, height)

# Compute the background models for each camera
background_models = {cam: compute_gaussian_model(resolution, vid=background_videos[cam]) for cam in cams}

# Get the silhouette frames for each camera
silhouettes = {cam: background_subtraction(background_models[cam], foreground_videos[cam], resolution) for cam in cams}

# TEST function to check if silouette frames are stored properly
def display_video_frames(frames, fps=30):
    for frame in frames:
        cv.imshow("Video", frame)
        
        if cv.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

# TEST video frames
display_video_frames(silhouettes["cam4"])

# Given a silhouettes array, frame number, and pixel coordinate, check if the pixel is white
def pixel_is_white(silhouettes, frame_number, y, x):
    
    frame = silhouettes[frame_number]

    # Return true if the pixel is white
    if frame[y, x] == 255:
        return True
    else:
        return False

# TEST the pixel values
pixel_value = pixel_is_white(silhouettes["cam4"], 10, 200, 250)
print(pixel_value)

'''
# Pseudocode for voxel iteration
def iterate_through_voxels(grid):
    for voxel in grid:
        for cam in cams:
            project_point(voxel_centerpoint)
            if NOT pixel_is_white(silhouettes[cam], frame_num, y, x)
                break

'''