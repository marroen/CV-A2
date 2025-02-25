
import cv2 as cv
import numpy as np

from background_subtraction import compute_gaussian_model
from background_subtraction import background_subtraction
from calibration_processing import load_config

# Background and foregrond videos
background_vid = cv.VideoCapture("data/cam2/background.avi")
foreground_vid = cv.VideoCapture("data/cam2/video.avi")

# Video resolution
width = 644
height = 486
resolution = (width, height)

# Compute the background model for cam1
#cam1_background_model = compute_gaussian_model(resolution, vid=background_vid)

# Use the cam1 background model for background subtraction on the foreground video
#background_subtraction(cam1_background_model, foreground_vid, resolution)

import numpy as np

def generate_voxel_corners():
    # Chessboard-aligned parameters (matches calibration pattern)
    voxel_size = 115  # mm, same as chessboard square size
    x_voxels = 8      # Match chessboard columns (8 squares in x)
    y_voxels = 6      # Match chessboard rows (6 squares in y)
    z_voxels = 3      # Vertical layers
    
    # Generate corner coordinates
    x = np.arange(0, (x_voxels+1)*voxel_size, voxel_size)
    y = np.arange(0, (y_voxels+1)*voxel_size, voxel_size)
    z = np.arange(0, (z_voxels+1)*voxel_size, voxel_size)
    
    # Create 3D grid of corner points
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

def generate_voxel_edges(corners):
    """Generate line segments between adjacent corners"""
    # Reshape to grid dimensions
    grid_shape = (len(np.unique(corners[:,0])), 
                 len(np.unique(corners[:,1])),
                 len(np.unique(corners[:,2])))
    grid = corners.reshape(grid_shape[0], grid_shape[1], grid_shape[2], 3)
    
    edges = []
    # Iterate through all possible edges
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                # X-direction edges
                if i < grid_shape[0]-1:
                    edges.append((grid[i,j,k], grid[i+1,j,k]))
                # Y-direction edges
                if j < grid_shape[1]-1:
                    edges.append((grid[i,j,k], grid[i,j+1,k]))
                # Z-direction edges
                if k < grid_shape[2]-1:
                    edges.append((grid[i,j,k], grid[i,j,k+1]))
    
    return np.array(edges, dtype=np.float32)

# World-space voxel grid (generated once)
WORLD_VOXEL_CORNERS = None
WORLD_VOXEL_EDGES = None

def initialize_world_voxel_grid():
    """Create world-space grid with origin at chessboard's bottom-left corner"""
    global WORLD_VOXEL_CORNERS, WORLD_VOXEL_EDGES
    
    voxel_size = 115  # mm, same as chessboard square size
    x_voxels = 8      # Match chessboard columns (X-axis)
    y_voxels = 6      # Match chessboard rows (Y-axis)
    z_voxels = 3      # Vertical layers (Z-axis)

    # X: Left to right (0 → 8 squares)
    x = np.arange(0, (x_voxels+1)*voxel_size, voxel_size)
    
    # Y: Bottom to top (690mm → 0mm in steps of -115mm)
    y = np.arange(y_voxels * voxel_size, -voxel_size, -voxel_size)
    
    # Z: Ground up (0 → 3 layers)
    z = np.arange(0, (z_voxels+1)*voxel_size, voxel_size)

    # Create grid with origin at (0,690,0) = bottom-left
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    WORLD_VOXEL_CORNERS = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    WORLD_VOXEL_EDGES = generate_voxel_edges(WORLD_VOXEL_CORNERS)

def project_voxel_grid():
    # Force reinitialize grid
    global WORLD_VOXEL_CORNERS, WORLD_VOXEL_EDGES
    WORLD_VOXEL_CORNERS = None
    WORLD_VOXEL_EDGES = None
    initialize_world_voxel_grid()
    
    # Load calibration data
    calib_data = load_config()
    
    # Create list of camera IDs
    cameras = [1, 2, 3, 4]
    
    for cam_id in cameras:
        # Get camera calibration
        cam_calib = calib_data.get(cam_id)
        if not cam_calib:
            print(f"No calibration data for camera {cam_id}")
            continue
            
        # Load checkerboard image
        img = cv.imread(f'data/cam{cam_id}/checkerboard.jpg')
        if img is None:
            print(f"Checkerboard image not found for camera {cam_id}")
            continue
            
        # Project edges
        edge_points = WORLD_VOXEL_EDGES.reshape(-1, 3)
        projected, _ = cv.projectPoints(edge_points,
                                      cam_calib['rvec'],
                                      cam_calib['tvec'],
                                      cam_calib['matrix'],
                                      cam_calib['dist_coef'])
        
        # Draw edges
        lines = projected.reshape(-1, 2, 2).astype(np.int32)
        for line in lines:
            (x1, y1), (x2, y2) = line[0], line[1]
            if all(0 <= x < img.shape[1] and 0 <= y < img.shape[0] 
                   for x, y in [(x1, y1), (x2, y2)]):
                cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Show and wait for space
        cv.imshow(f'Camera {cam_id} - Press SPACE to continue', img)
        while True:
            key = cv.waitKey(0) & 0xFF
            if key == 32:  # ASCII code for space
                cv.destroyAllWindows()
                break
            elif key == 27:  # ESC to exit
                cv.destroyAllWindows()
                return

    # Final message
    print("All camera projections completed")

# Example usage
project_voxel_grid()