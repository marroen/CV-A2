import cv2 as cv
import os
import numpy as np

# Class to store calibration data and project 3D points to 2D image points
class CalibrationInstance:
    def __init__(self, ret, matrix, distortion_coef, rotation_vecs, translation_vecs):
        self.ret = ret
        self.matrix = matrix
        self.distortion_coef = distortion_coef
        self.rotation_vecs = rotation_vecs
        self.translation_vecs = translation_vecs

    def project_points(self, object_points, pose_index=0):
        """
        Projects 3D object points to 2D image points using the specified calibration pose.

        :param object_points: np.array of 3D points to project.
        :param pose_index: index of the pose (rvec/tvec pair) to use.
        :return: projected 2D image points.
        """
        if not self.rotation_vecs or not self.translation_vecs:
            raise ValueError("No extrinsic parameters available.")

        imgpts, _ = cv.projectPoints(object_points,
                                      self.rotation_vecs[pose_index],
                                      self.translation_vecs[pose_index],
                                      self.matrix,
                                      self.distortion_coef)
        return imgpts

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

def get_calibration_paths(base_dir="data"):
    """Get all calibration file paths for cameras in base directory"""
    calib_paths = []
    for cam_dir in os.listdir(base_dir):
        cam_path = os.path.join(base_dir, cam_dir)
        if os.path.isdir(cam_path):
            calib_file = os.path.join(cam_path, "calibration.xml")
            if os.path.exists(calib_file):
                calib_paths.append(calib_file)
    return calib_paths