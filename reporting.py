import cv2 as cv
import numpy as np
import xml.etree.ElementTree as ET

# Retrieves data from the xml file
def parse_matrix(xml_element):
    data = list(map(float, xml_element.find("data").text.split()))
    rows = int(xml_element.find("rows").text)
    cols = int(xml_element.find("cols").text)

    return np.array(data, dtype=np.float64).reshape(rows, cols)

# Load XML file
xml_file = "config.xml"
tree = ET.parse(xml_file)
root = tree.getroot()

# Iterate over cameras
for camera in root.findall("camera1") + root.findall("camera2") + root.findall("camera3") + root.findall("camera4"):
    camera_name = camera.tag

    # Get rvec and tvec
    rvec = parse_matrix(camera.find("rotation_vector"))
    tvec = parse_matrix(camera.find("translation_vector"))

    # Convert rvec to rotation matrix
    R, _ = cv.Rodrigues(rvec)

    # Print results
    print(f"\n{camera_name}:")
    print("R:")
    print(np.round(R, 6))
    print("tvec:")
    print(np.round(tvec, 6))
