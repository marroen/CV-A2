
import cv2 as cv
import numpy as np

# Background and foregrond videos
background_vid = cv.VideoCapture("data/cam1/background.avi")
foreground_vid = cv.VideoCapture("data/cam1/video.avi")

# Video resolution
width = 644
height = 486
resolution = (width, height)

# Create separate Gaussian models for each camera
# Stores the mean and variance of HSV values for each pixel in the background
# Eg. for pixel (y,x): [y, x, [mean_H, mean_S, mean_V, var_H, var_S, var_V]]
cam1_background_model = np.zeros((height, width, 6), dtype=np.float32)
cam2_background_model = np.zeros((height, width, 6), dtype=np.float32)
cam3_background_model = np.zeros((height, width, 6), dtype=np.float32)
cam4_background_model = np.zeros((height, width, 6), dtype=np.float32)

# For each background model, initialize variance for H, S, V to small value (to avoid dividing by 0 later)
for model in [cam1_background_model, cam2_background_model, cam3_background_model, cam4_background_model]:
    model[:, :, 3:] = 1e-6

# Computes the gaussian distribution of the given video and stores this in the given background model
def computeBackgroundModel(background_model, vid, resolution):

    width, height = resolution
    hsv_frames = []

    # Iterate through each frame
    frame_count = 0
    while True:

        success, frame = vid.read()  # Reads the next frame
        if not success:
            cv.imshow("Frame", last_frame)
            key = cv.waitKey(0)  # Waits for keypress
            if key == 27:  # Halt if ESC pressed
                exit()
            break  # Break when video ends

        # Convert frame to HSV
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Append frame to array
        hsv_frames.append(hsv_frame.copy())

        print(f"Processing frame {frame_count}")

        frame_count += 1
        last_frame = frame

    # Convert frames to numpy array
    # Eg. [num_frames, height, width, [H, S, V]]
    hsv_frames = np.array(hsv_frames)

    # Iterates through each HSV value of each pixel of each frame, and calculates the mean and variance
    # TODO: find other algs for variation https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    cam1_background_model[:, :, :3] = np.mean(hsv_frames, axis=0)
    cam1_background_model[:, :, 3:] = np.var(hsv_frames, axis=0)

    vid.release()
    print(f"\nProcessed {frame_count} frames")

# Performes background subtraction on the given video compared to the given background model
def backgroundSubtraction(background_model, vid, resolution):

    width, height = resolution
    hsv_frames = []

    # Iterate through each frame
    frame_count = 0
    while True:

        success, frame = vid.read()  # Reads the next frame
        if not success:
            break  # Break when video ends

        # Convert frame to HSV
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        threshold = 20  # Number of standard deviations

        # Difference of frame HSV and background mean HSV
        # Eg. (foreground mean HSV [05, 20, 50]) - (background mean HSV [50, 50, 05]) = [45, 30, 0]
        mean_diff = hsv_frame - background_model[:, :, :3]

        # Finds the distance between the foreground frame value and the background model distribution
        mahalanobis_dist = (
            (mean_diff[:, :, 0] ** 2) / (background_model[:, :, 0] + 1e-6) + # H
            (mean_diff[:, :, 1] ** 2) / (background_model[:, :, 1] + 1e-6) + # S
            (mean_diff[:, :, 2] ** 2) / (background_model[:, :, 2] + 1e-6)   # V
        )

        # Initializes foreground frame with all 0's
        foreground_frame = np.zeros(mahalanobis_dist.shape, dtype=np.uint8)
        foreground_frame[mahalanobis_dist > threshold] = 255 # Set any pixel outide threshold to white

        cv.imshow("Foreground", foreground_frame)
        key = cv.waitKey(0)  # Waits for keypress
        if key == 27:  # Halt if ESC pressed
            exit()

        # Append frame to array
        hsv_frames.append(hsv_frame.copy())

        print(f"Processing frame {frame_count}")

        frame_count += 1

    vid.release()
    print(f"\nProcessed {frame_count} frames")


# Compute the background model for cam1
computeBackgroundModel(cam1_background_model, background_vid, resolution)

# Use the cam1 background model for background subtraction on the foreground video
backgroundSubtraction(cam1_background_model, foreground_vid, resolution)


# TEST to see if the background model worked
# Print mean and variance for the test pixels
test_pixels = [(100, 100), (200, 400), (21, 364)]
for y, x in test_pixels:
    mean_H, mean_S, mean_V = cam1_background_model[y, x, :3]
    var_H, var_S, var_V = cam1_background_model[y, x, 3:]

    print(f"\npixel ({y}, {x}):")
    print(f"  HSV mean: ({mean_H:.3f}, {mean_S:.3f}, {mean_V:.3f})")
    print(f"  HSV variance: ({var_H:.3f}, {var_S:.3f}, {var_V:.3f})")
    print("-" * 60)