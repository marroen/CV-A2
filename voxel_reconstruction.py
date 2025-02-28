
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

display_video_frames(silhouettes["cam4"])

# Initialize 100x100x100 voxel grid=
GRID_SIZE = 100    # Number of voxels per dimension
VOXEL_SIZE = 0.8   # Physical size of each voxel cube
SPACING = 0.2      # Space between voxels
PADDING = 10.0     # Padding around grid for visualization

# Calculate total grid dimensions
# Calculate grid dimensions
total_offset = (GRID_SIZE - 1) * (VOXEL_SIZE + SPACING) / 2
origin = [-total_offset - PADDING, 
          -total_offset - PADDING, 
          -total_offset - PADDING]

# Generate ALL voxel positions (regardless of occupancy)
voxel_positions = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, 3), dtype=np.float32)

for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        for k in range(GRID_SIZE):
            x = origin[0] + i * (VOXEL_SIZE + SPACING)
            y = origin[1] + j * (VOXEL_SIZE + SPACING)
            z = origin[2] + k * (VOXEL_SIZE + SPACING)
            voxel_positions[i, j, k] = [x, y, z]

# Convert to flat array of positions (shape: [1000000, 3])
all_voxel_centers = voxel_positions.reshape(-1, 3)

def voxel_grid():
    print("Voxel Grid")
    P = all_voxel_centers
    print(silhouettes['cam1'][0])
    print(silhouettes['cam1'][0].shape)
    F = silhouettes['cam1']
    #for p in P:
        #print(p)
voxel_grid()