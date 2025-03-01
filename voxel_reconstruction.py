
import cv2 as cv
import numpy as np

from background_subtraction import compute_gaussian_model
from background_subtraction import background_subtraction
from calibration_processing import load_config

# Store videos in dictionary for each camera
cams = ["cam1", "cam2", "cam3", "cam4"]
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

# Plays a video of the given frames
def display_video_frames(frames, fps=120):

    for frame in frames:
        cv.imshow("Video", frame)
        
        if cv.waitKey(int(1000 / fps)) & 0xFF == 27:
            break

    cv.destroyAllWindows()

# Plays a short video of each silhouette view
for cam in cams:
    display_video_frames(silhouettes[cam])

# Given a silhouettes array and pixel coordinate, check if the pixel is white
def pixel_is_white(silhouettes, x, y):
    
    for i in range(100):
        frame = silhouettes[i]
        # Continue if pixel is ON
        if frame[y, x] == 255:
            continue
        # Return False if the pixel is OFF
        else:
            return False
    return True

    

# Define voxel data type
voxel_dtype = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('occupied', np.bool_)
])

# Grid dimensions (voxel counts)
GRID_WIDTH = 10    # X-axis (left-right)
GRID_DEPTH = 12    # Y-axis (forward-back)
GRID_HEIGHT = 20   # Z-axis (vertical)
VOXEL_SIZE = 100.0    # Size of each voxel cube

GRID_WIDTH = 15    # X-axis (left-right)
GRID_DEPTH = 18    # Y-axis (forward-back)
GRID_HEIGHT = 30   # Z-axis (vertical)
VOXEL_SIZE = 60.0    # Size of each voxel cube

GRID_WIDTH = 150    # X-axis (left-right)
GRID_DEPTH = 175    # Y-axis (forward-back)
GRID_HEIGHT = 300   # Z-axis (vertical)
VOXEL_SIZE = 6.0    # Size of each voxel cube

GRID_WIDTH = 29    # X-axis (left-right)
GRID_DEPTH = 29    # Y-axis (forward-back)
GRID_HEIGHT = 39   # Z-axis (vertical)
VOXEL_SIZE = 60.0    # Size of each voxel cube

GRID_WIDTH = 15    # X-axis (left-right)
GRID_DEPTH = 18    # Y-axis (forward-back)
GRID_HEIGHT = 30   # Z-axis (vertical)
VOXEL_SIZE = 60.0    # Size of each voxel cube
ANCHOR = np.array([0.0, 0.0, 0.0])  # World coordinates of bottom-front-left corner

# Initialize voxel grid with proper dimensions
voxels = np.zeros(GRID_WIDTH * GRID_DEPTH * GRID_HEIGHT, dtype=voxel_dtype)

# Draw voxel grid, extending towards camera, from world origin
index = 0
for i in range(GRID_WIDTH):
    for j in range(GRID_DEPTH):
        for k in range(GRID_HEIGHT):
            # Calculate positions relative to anchor
            x = ANCHOR[0] + i * VOXEL_SIZE  # X decreases (left in camera view)
            y = ANCHOR[1] + j * VOXEL_SIZE  # Y increases (forward in camera view)
            z = ANCHOR[2] - k * VOXEL_SIZE  # Z increases (upward from ground)
            
            voxels[index] = (x, y, z, False)
            index += 1


# Usage example with camera alignment
calib_data = load_config()

'''
# TEST the voxel grid (VISUALIZATION) CAM1
cam_calib = calib_data.get(1)
only_voxels = np.stack([voxels['x'], voxels['y'], voxels['z']], axis=1)
projs, _ = cv.projectPoints(only_voxels.astype(np.float32), cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])

projs = projs.reshape(-1, 2)


# Draw voxel grid for debug
img = cv.imread('data/cam1/checkerboard.jpg')
output = img.copy()
for x, y in projs:  # Directly unpack x,y
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        cv.circle(output, (int(x), int(y)), 
                    2, (0, 255, 0), -1)

cv.imshow('img', output)
cv.waitKey(0)
cv.destroyAllWindows()

# TEST the voxel grid (VISUALIZATION) CAM2
cam_calib = calib_data.get(2)
only_voxels = np.stack([voxels['x'], voxels['y'], voxels['z']], axis=1)
projs, _ = cv.projectPoints(only_voxels.astype(np.float32), cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])

projs = projs.reshape(-1, 2)


# Draw voxel grid for debug
img = cv.imread('data/cam2/checkerboard.jpg')
output = img.copy()
for x, y in projs:  # Directly unpack x,y
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        cv.circle(output, (int(x), int(y)), 
                    2, (0, 255, 0), -1)

cv.imshow('img', output)
cv.waitKey(0)
cv.destroyAllWindows()

# TEST the voxel grid (VISUALIZATION) CAM3
cam_calib = calib_data.get(3)
only_voxels = np.stack([voxels['x'], voxels['y'], voxels['z']], axis=1)
projs, _ = cv.projectPoints(only_voxels.astype(np.float32), cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])

projs = projs.reshape(-1, 2)


# Draw voxel grid for debug
img = cv.imread('data/cam3/checkerboard.jpg')
output = img.copy()
for x, y in projs:  # Directly unpack x,y
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        cv.circle(output, (int(x), int(y)), 
                    2, (0, 255, 0), -1)

cv.imshow('img', output)
cv.waitKey(0)
cv.destroyAllWindows()

# TEST the voxel grid (VISUALIZATION) CAM4
cam_calib = calib_data.get(4)
only_voxels = np.stack([voxels['x'], voxels['y'], voxels['z']], axis=1)
projs, _ = cv.projectPoints(only_voxels.astype(np.float32), cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])

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
cv.destroyAllWindows()'''


def voxel_grid():
    print("Voxel Grid")
    P = voxels
    #print(silhouettes['cam1'][0])
    #print(silhouettes['cam1'][0].shape)
    F1 = silhouettes['cam1']
    F2 = silhouettes['cam2']
    F3 = silhouettes['cam3']
    F4 = silhouettes['cam4']
    for p in P:
        num_ons = 0
        cam_calib = calib_data.get(1)
        p2 = np.array([p['x'], p['y'], p['z']], dtype=np.float32)
        projected, _ = cv.projectPoints(p2, cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])
        projected = projected.flatten()
        #print(projected[0])
        #print(projected)
        #print(type(projected))
        x_c1 = round(projected[0])
        y_c1 = round(projected[1])
        if (pixel_is_white(F1, x_c1, y_c1)):
            num_ons += 1
            cam_calib = calib_data.get(2)
            projected, _ = cv.projectPoints(p2, cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])
            projected = projected.flatten()
            x_c2 = round(projected[0])
            y_c2 = round(projected[1])
            if (pixel_is_white(F2, x_c2, y_c2)):
                num_ons += 1
                cam_calib = calib_data.get(3)
                projected, _ = cv.projectPoints(p2, cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])
                projected = projected.flatten()
                x_c3 = round(projected[0])
                y_c3 = round(projected[1])
                if (pixel_is_white(F3, x_c3, y_c3)):
                    num_ons += 1
                    cam_calib = calib_data.get(4)
                    projected, _ = cv.projectPoints(p2, cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])
                    projected = projected.flatten()
                    x_c4 = round(projected[0])
                    y_c4 = round(projected[1])
                    if (pixel_is_white(F4, x_c4, y_c4)):
                        num_ons += 1
                        print("all was on")
        if num_ons >= 3:
            p['occupied'] = True
    return P

P = voxel_grid()

# TEST the voxel grid (VISUALIZATION)
cam_calib = calib_data.get(2)
only_voxels = np.stack([voxels['x'], voxels['y'], voxels['z']], axis=1)
only_voxels_on = []
for p in P:
    if p['occupied']:
        only_voxels_on.append([p['x'], p['y'], p['z']])
only_voxels_on = np.array(only_voxels_on)
projs, _ = cv.projectPoints(only_voxels_on.astype(np.float32), cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])

projs = projs.reshape(-1, 2)


# Draw voxel grid for debug
img = cv.imread('data/cam2/checkerboard.jpg')
output = img.copy()
for x, y in projs:  # Directly unpack x,y
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        cv.circle(output, (int(x), int(y)), 
                    2, (0, 255, 0), -1)

cv.imshow('img', output)
cv.waitKey(0)
cv.destroyAllWindows()