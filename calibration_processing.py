import cv2 as cv
import os
import numpy as np

# A1 imports
import project
from CalibrationInstance import CalibrationInstance

# Get camera intrinsics with code from A1, and save to calibration files per camera
def get_intrinsics():
    rows = 6
    cols = 8
    stride = 115
    grid = (cols, rows)
    
    # Get all points from each camera
    cam1_points = project.get_all_points(25, rows, cols, stride, video_path='data/cam1/intrinsics.avi', calibration=True)
    cam2_points = project.get_all_points(25, rows, cols, stride, video_path='data/cam2/intrinsics.avi', calibration=True)
    cam3_points = project.get_all_points(25, rows, cols, stride, video_path='data/cam3/intrinsics.avi', calibration=True)
    cam4_points = project.get_all_points(25, rows, cols, stride, video_path='data/cam4/intrinsics.avi', calibration=True)

    # Calibrate each camera and save to file
    cam1_calib = project.calibrate_camera(cam1_points)
    save_calibration('data/cam1', cam1_calib)
    cam2_calib = project.calibrate_camera(cam2_points)
    save_calibration('data/cam2', cam2_calib)
    cam3_calib = project.calibrate_camera(cam3_points)
    save_calibration('data/cam3', cam3_calib)
    cam4_calib = project.calibrate_camera(cam4_points)
    save_calibration('data/cam4', cam4_calib)

    # Return calibration objects instead of saving
    return [
        project.calibrate_camera(cam1_points),
        project.calibrate_camera(cam2_points),
        project.calibrate_camera(cam3_points),
        project.calibrate_camera(cam4_points)
    ]

    # if calibration exists, load it TODO

def get_extrinsics(cam, objp):
    """Get extrinsic parameters for a specific camera using its checkerboard image"""
    print(f"Processing camera {cam}")

    rows = 6
    cols = 8
    grid = (cols, rows)
    
    # Load intrinsic calibration
    cam_calib = load_calibration(f'data/cam{cam}')
    
    # Load checkerboard image
    frame = cv.imread(f'data/cam{cam}/checkerboard.jpg')
    if frame is None:
        raise FileNotFoundError(f"Checkerboard image not found for camera {cam}")
    
    # Corner detection and refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    '''
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (8, 6), None)
    
    if ret:
        refined_corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        ret, rvec, tvec = cv.solvePnP(objp, refined_corners, 
                                     cam_calib.matrix, cam_calib.distortion_coef)
        return rvec, tvec
    else:
        test_points = project.get_points(25, f'data/cam{cam}/checkerboard.jpg', 0)
        img = test_points[3] # Same img (halved) from where points are initially extracted
        preprocessed = project.preprocessing(img)
        refined_corners = cv.cornerSubPix(preprocessed, test_points[1], (11,11), (-1,-1), criteria)
        _, rvec, tvec = cv.solvePnP(objp, refined_corners, cam_calib.matrix, cam_calib.distortion_coef)
        return rvec, tvec'''
    
    preprocessed = project.preprocessing(frame)
    interpolated_corners = project.manual_check(preprocessed)

    # Use 2D ideal grid corners for homography
    # Define 4 corners of the ideal 7x7 grid (2D!)
    ideal_manual_points = np.array([
        [0, 0],    # top-left
        [grid[0]-1, 0],    # top-right (7x7 grid has 0-6 in x)
        [grid[0]-1, grid[1]-1],    # bottom-right
        [0, grid[1]-1]     # bottom-left
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Compute homography
    H, _ = cv.findHomography(ideal_manual_points, interpolated_corners)
    x, y = np.meshgrid(np.arange(grid[0]), np.arange(grid[1]))
    ideal_grid = np.float32(np.vstack([x.ravel(), y.ravel()]).T).reshape(-1, 1, 2)
    interpolated_corners = cv.perspectiveTransform(ideal_grid, H).reshape(-1, 2)

    # Draw and display the corners
    draw_img = frame.copy()
    cv.drawChessboardCorners(draw_img, grid, interpolated_corners, True)
    cv.imshow('img', draw_img)
    cv.waitKey(500)
    
    refined_corners = cv.cornerSubPix(preprocessed, interpolated_corners, (11,11), (-1,-1), criteria)
    _, rvec, tvec = cv.solvePnP(objp, refined_corners, cam_calib.matrix, cam_calib.distortion_coef)
    return rvec, tvec
    
# After running the calibration, save the calibration data to a camera-specific XML file
def save_calibration(camera_dir, calibration):
    """Save calibration data to camera-specific XML file"""
    # Create camera directory if needed
    os.makedirs(camera_dir, exist_ok=True)
    
    fs = cv.FileStorage(os.path.join(camera_dir, "calibration.xml"), cv.FILE_STORAGE_WRITE)
    
    # Write basic parameters
    fs.write("ret", calibration.ret)
    fs.write("camera_matrix", calibration.matrix)
    fs.write("distortion_coefficients", calibration.distortion_coef)
    
    # Write rotation vectors as a sequence
    fs.startWriteStruct("rotation_vectors", cv.FileNode_SEQ)
    for rvec in calibration.rotation_vecs:
        fs.write("", rvec)  # Empty name for sequence elements
    fs.endWriteStruct()
    
    # Write translation vectors as a sequence
    fs.startWriteStruct("translation_vectors", cv.FileNode_SEQ)
    for tvec in calibration.translation_vecs:
        fs.write("", tvec)  # Empty name for sequence elements
    fs.endWriteStruct()
    
    fs.release()
    return os.path.join(camera_dir, "calibration.xml")

# Load calibration data from camera-specific XML file
def load_calibration(camera_dir):
    """Load calibration data from camera-specific XML file"""
    calib_path = os.path.join(camera_dir, "calibration.xml")
    
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"No calibration file found in {camera_dir}")
    
    fs = cv.FileStorage(calib_path, cv.FILE_STORAGE_READ)
    
    # Read parameters
    ret = fs.getNode("ret").real()
    matrix = fs.getNode("camera_matrix").mat()
    distortion_coef = fs.getNode("distortion_coefficients").mat().flatten()
    
    # Read vectors
    rotation_node = fs.getNode("rotation_vectors")
    rotation_vecs = [rotation_node.at(i).mat() for i in range(rotation_node.size())]
    
    translation_node = fs.getNode("translation_vectors")
    translation_vecs = [translation_node.at(i).mat() for i in range(translation_node.size())]
    
    fs.release()
    
    return CalibrationInstance(
        ret=ret,
        matrix=matrix,
        distortion_coef=distortion_coef,
        rotation_vecs=rotation_vecs,
        translation_vecs=translation_vecs
    )

def save_combined_config():
    """Save all camera data to a single config.xml, generating missing calibrations"""
    fs = cv.FileStorage("config.xml", cv.FILE_STORAGE_WRITE)
    
    # Generate 3D object points (same as calibration pattern)
    _, objp, _, _ = project.initialize_points(6, 8, 115)

    for cam in range(1, 5):
        cam_dir = f'data/cam{cam}'
        calib_path = os.path.join(cam_dir, "calibration.xml")
        
        # Generate intrinsic calibration if missing
        if not os.path.exists(calib_path):
            print(f"Generating intrinsics for camera {cam}")
            video_path = os.path.join(cam_dir, "intrinsics.avi")
            
            # Get calibration points from video
            points = project.get_all_points(
                25, 6, 8, 115, 
                video_path=video_path,
                calibration=True
            )
            
            # Calibrate and save
            calib = project.calibrate_camera(points)
            save_calibration(cam_dir, calib)

        # Now load existing/new calibration
        cam_calib = load_calibration(cam_dir)
        
        # Get extrinsics (will auto-detect from checkerboard.jpg)
        try:
            rvec, tvec = get_extrinsics(cam, objp)
        except Exception as e:
            print(f"Error getting extrinsics for camera {cam}: {str(e)}")
            continue

        # Write to combined config
        fs.startWriteStruct(f"camera{cam}", cv.FILE_NODE_MAP)
        fs.write("camera_matrix", cam_calib.matrix)
        fs.write("distortion_coefficients", cam_calib.distortion_coef)
        fs.write("rotation_vector", rvec)
        fs.write("translation_vector", tvec)
        fs.endWriteStruct()
    
    fs.release()
    print("Combined config saved to config.xml")

def load_config(config_path="config.xml"):
  """Load combined calibration config for all cameras"""
  if not os.path.exists(config_path):
      raise FileNotFoundError(f"Config file {config_path} not found")

  fs = cv.FileStorage(config_path, cv.FILE_STORAGE_READ)
  cameras = {}

  for cam_id in range(1, 5):  # For cameras 1-4
      node = fs.getNode(f"camera{cam_id}")
      if node.empty():
          continue  # Skip missing cameras

      cameras[cam_id] = {
          "matrix": node.getNode("camera_matrix").mat(),
          "dist_coef": node.getNode("distortion_coefficients").mat().flatten(),
          "rvec": node.getNode("rotation_vector").mat(),
          "tvec": node.getNode("translation_vector").mat()
      }

  fs.release()
  return cameras