
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

# Plays a video of the given frames
def display_video_frames_fr(frames, fps=120):

    for i, frame in enumerate(frames):  # Add index tracking
        cv.imshow("Video", frame)
        
        # Pause at frame 50
        if i == 50:
            print("Paused at frame 50. Press any key to continue...")
            cv.waitKey(0)  # Wait indefinitely until a key is pressed
        
        # Continue normal playback
        if cv.waitKey(int(1000 / fps)) & 0xFF == 27:
            break

    cv.destroyAllWindows()


# Plays a short video of each silhouette view
for cam in cams:
    display_video_frames_fr(silhouettes[cam])

# Given a silhouettes array, frame number, and pixel coordinate, check if the pixel is white
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
    ('x', np.int32),
    ('y', np.int32),
    ('z', np.int32),
    ('occupied', np.bool_)
])

# Grid dimensions (voxel counts)
GRID_WIDTH = 150    # X-axis (left-right)
GRID_DEPTH = 175    # Y-axis (forward-back)
GRID_HEIGHT = 300   # Z-axis (vertical)
ANCHOR = np.array([0.0, 0.0, 0.0])  # World coordinates of bottom-front-left corner

# Voxel metrics (in meters)
VOXEL_SIZE = 6.0    # Size of each voxel cube
SPACING = 0.0       # No spacing between voxels

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
# TEST the voxel grid (VISUALIZATION)
cam_calib = calib_data.get(4)
only_voxels = np.stack([voxels['x'], voxels['y'], voxels['z']], axis=1)
projs, _ = cv.projectPoints(only_voxels.astype(np.float32), cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])
#projs = projs.squeeze()

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
            cam_calib = calib_data.get(2)
            projected, _ = cv.projectPoints(p2, cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])
            projected = projected.flatten()
            x_c2 = round(projected[0])
            y_c2 = round(projected[1])
            if (pixel_is_white(F2, x_c2, y_c2)):
                cam_calib = calib_data.get(3)
                projected, _ = cv.projectPoints(p2, cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])
                projected = projected.flatten()
                x_c3 = round(projected[0])
                y_c3 = round(projected[1])
                if (pixel_is_white(F3, x_c3, y_c3)):
                    cam_calib = calib_data.get(4)
                    projected, _ = cv.projectPoints(p2, cam_calib['rvec'], cam_calib['tvec'], cam_calib['matrix'], cam_calib['dist_coef'])
                    projected = projected.flatten()
                    x_c4 = round(projected[0])
                    y_c4 = round(projected[1])
                    if (pixel_is_white(F4, x_c4, y_c4)):
                        p['occupied'] = True
                        print("was on")

voxel_grid()