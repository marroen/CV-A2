
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

# Given a silhouettes array, frame number, and pixel coordinate, check if the pixel is white
def pixel_is_white(silhouettes, frame_index, x, y):
    
    frame = silhouettes[frame_index]

    # Return true if the pixel is white
    if frame[y, x] == 255:
        return True
    else:
        return False

display_video_frames(silhouettes["cam4"])

# Define voxel data type
voxel_dtype = np.dtype([
    ('x', np.int32),
    ('y', np.int32),
    ('z', np.int32),
    ('occupied', np.bool_)
])

# Initialize voxel grid parameters
GRID_WIDTH = 200   # X-axis (left-right)
GRID_DEPTH = 150    # Y-axis (front-back)
GRID_HEIGHT = 300   # Z-axis (vertical)
VOXEL_SIZE = 6.0   # Physical size of each voxel cube
SPACING = 6.0      # Space between voxels
PADDING = 10.0     # Padding around grid for visualization

# Calculate grid origins PER AXIS (centered with padding)
def calculate_origin(grid_dim, voxel_size, spacing, padding):
    total_size = grid_dim * (voxel_size + spacing) - spacing
    return -total_size/2 - padding

# X-axis (width)
origin_x = calculate_origin(GRID_WIDTH, VOXEL_SIZE, SPACING, 20.0)
# Y-axis (depth)
origin_y = calculate_origin(GRID_DEPTH, VOXEL_SIZE, SPACING, 15.0)
# Z-axis (height) - less padding at bottom for "ground"
origin_z = calculate_origin(GRID_HEIGHT, VOXEL_SIZE, SPACING, 30.0) 

# Initialize voxel grid with proper dimensions
voxels = np.zeros(GRID_WIDTH * GRID_DEPTH * GRID_HEIGHT, dtype=voxel_dtype)

# Populate grid with human-centric coordinates
index = 0
step = VOXEL_SIZE + SPACING
for i in range(GRID_WIDTH):
    for j in range(GRID_DEPTH):
        for k in range(GRID_HEIGHT):
            voxels[index] = (
                origin_x + i * step,  # X-coordinate
                origin_y + j * step,  # Y-coordinate
                origin_z + k * step,  # Z-coordinate (vertical)
                False
            )
            index += 1

# Usage example with camera alignment
calib_data = load_config()

'''
# TEST the voxel grid
cam_calib = calib_data.get(4)
only_voxels = np.stack([voxels['x'], voxels['y'], voxels['z']], axis=1)
projs, _ = cv.projectPoints(only_voxels.astype(np.float32), cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])
projs = projs.squeeze()

# Calculate depths
R, _ = cv.Rodrigues(cam_calib['rvec'])
camera_coords = (R @ only_voxels.T).T + cam_calib['tvec'].T
depths = camera_coords[:, 2]

projs = projs.reshape(-1, 2)


# Draw voxel grid for debug
img = cv.imread('data/cam4/checkerboard.jpg')
output = img.copy()
for x, y in projs:  # Directly unpack x,y
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        cv.circle(output, (int(x), int(y)), 
                    2, (0, 255, 0), -1)

cv.imshow('img', output)
cv.waitKey(0)
cv.destroyAllWindows()
'''

def voxel_grid():
    print("Voxel Grid")
    P = voxels
    #print(silhouettes['cam1'][0])
    #print(silhouettes['cam1'][0].shape)
    F = silhouettes['cam1']
    for p in P:
        cam_calib = calib_data.get(1)
        p2 = np.array([p['x'], p['y'], p['z']], dtype=np.float32)
        projected, _ = cv.projectPoints(p2, cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])
        projected = projected.flatten()
        #print(projected[0])
        #print(projected)
        #print(type(projected))
        x_c1 = round(projected[0])
        y_c1 = round(projected[1])
        if (pixel_is_white(silhouettes['cam1'], 0, x_c1, y_c1)):
            cam_calib = calib_data.get(2)
            projected, _ = cv.projectPoints(p2, cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])
            projected = projected.flatten()
            x_c2 = round(projected[0])
            y_c2 = round(projected[1])
            if (pixel_is_white(silhouettes['cam2'], 0, x_c2, y_c2)):
                cam_calib = calib_data.get(3)
                projected, _ = cv.projectPoints(p2, cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])
                projected = projected.flatten()
                x_c3 = round(projected[0])
                y_c3 = round(projected[1])
                if (pixel_is_white(silhouettes['cam3'], 0, x_c3, y_c3)):
                    cam_calib = calib_data.get(4)
                    projected, _ = cv.projectPoints(p2, cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])
                    projected = projected.flatten()
                    x_c4 = round(projected[0])
                    y_c4 = round(projected[1])
                    if (pixel_is_white(silhouettes['cam4'], 0, x_c4, y_c4)):
                        p['occupied'] = True

voxel_grid()