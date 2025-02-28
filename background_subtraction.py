
import cv2 as cv
import numpy as np

# Computes the gaussian distribution given EITHER a video OR an hsv_frames array and returns the gaussian model
def compute_gaussian_model(resolution, vid=None, hsv_frames=None):

    width, height = resolution

    # Stores the mean and variance of HSV values for each pixel in the frame
    # Eg. for pixel (y,x): [y, x, [mean_H, mean_S, mean_V, var_H, var_S, var_V]]
    gaussian_model = np.zeros((height, width, 6), dtype=np.float32)

    # Initialize variance for H, S, V to small value (to avoid dividing by 0 later)
    gaussian_model[:, :, 3:] = 1e-6

    # If passed a video, process video
    if hsv_frames is None:
        hsv_frames = []
        frame_count = 0

        # Iterate through each frame
        while True:

            success, frame = vid.read()  # Reads the next frame
            if not success:
                break  # Break when video ends

            hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV) # Convert frame to HSV
            hsv_frames.append(hsv_frame.copy()) # Append frame to array

            #print(f"compute_gaussian_model: Processing frame {frame_count}")
            frame_count += 1
        
        vid.release()
        print(f"compute_gaussian_model: Processed {frame_count} frames\n")

    # Convert frames to numpy array
    # Eg. [num_frames, height, width, [H, S, V]]
    hsv_frames = np.array(hsv_frames)

    # Iterates through each HSV value of each pixel of each frame, and calculates the mean and variance
    # TODO: find other algs for variation https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    gaussian_model[:, :, :3] = np.mean(hsv_frames, axis=0)
    gaussian_model[:, :, 3:] = np.var(hsv_frames, axis=0)

    return gaussian_model

# Performes background subtraction on the given video compared to the given background model
def background_subtraction(background_model, vid, resolution):

    width, height = resolution
    foreground_silhouettes = [] # Stores processed foreground silhouette frames

    threshold = 20  # TODO: automate number of standard deviations

    # Iterate through each frame
    frame_count = 0
    while True:

        success, frame = vid.read()  # Reads the next frame
        if not success:
            break  # Break when video ends
        elif frame_count >= 100:
            break # Break when 100 frames are processed

        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV) #Convert frame to HSV

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

        # Set any pixel outide threshold to white
        foreground_frame[mahalanobis_dist > threshold] = 255

        '''
        cv.imshow("Foreground", foreground_frame)
        key = cv.waitKey(0)  # Waits for keypress
        if key == 27:  # Halt if ESC pressed
            exit()
        '''

        foreground_silhouettes.append(foreground_frame.copy()) # Append frame to array

        #print(f"background_subtraction: Processing frame {frame_count}")
        frame_count += 1

    vid.release()
    print(f"background_subtraction: Processed {frame_count} frames\n")
    return(foreground_silhouettes)

# IN PROGRESS
# IDEA: get the gaussian distribution of the 5 frame buffer and find distance to background
# Didn't work very well
def background_subtraction_buffer(background_model, vid, resolution):

    width, height = resolution
    foreground_silhouettes = [] # Stores processed foreground silhouette frames
    buffer_size = 5

    threshold = 20  # TODO: automate number of standard deviations

    # Iterate through each frame
    frame_count = 0
    while True:

        success, frame = vid.read()  # Reads the next frame
        if not success:
            break  # Break when video ends
        elif frame_count >= 100:
            break # Break when 100 frames are processed

        # Keep only the last 5 frames in buffer
        if len(hsv_frames) > 5:
            hsv_frames.pop(0)

        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  #Convert frame to HSV

        # Difference of frame HSV and background mean HSV
        # Eg. (foreground mean HSV [05, 20, 50]) - (background mean HSV [50, 50, 05]) = [45, 30, 0]
        mean_diff = hsv_frame - background_model[:, :, :3]

        # Wait until we have at least 5 frames
        if len(hsv_frames) < 5:
            print(f"Skipping frame {frame_count} (need at least 5)")
            hsv_frames.append(hsv_frame.copy())
            frame_count += 1
            continue

        foreground_model = compute_gaussian_model(resolution, hsv_frames=np.array(hsv_frames))

        # Difference of frame HSV and background mean HSV
        # Eg. (foreground mean HSV [05, 20, 50]) - (background mean HSV [50, 50, 05]) = [45, 30, 0]
        mean_diff = foreground_model[:, :, :3] - background_model[:, :, :3]

        # Finds the distance between the foreground frame value and the background model distribution
        mahalanobis_dist = (
            (mean_diff[:, :, 0] ** 2) / (background_model[:, :, 0] + 1e-6) + # H
            (mean_diff[:, :, 1] ** 2) / (background_model[:, :, 1] + 1e-6) + # S
            (mean_diff[:, :, 2] ** 2) / (background_model[:, :, 2] + 1e-6)   # V
        )

        # Initializes foreground frame with all 0's
        foreground_frame = np.zeros(mahalanobis_dist.shape, dtype=np.uint8)

        # Set any pixel outide threshold to white
        foreground_frame[mahalanobis_dist > threshold] = 255

        '''
        cv.imshow("Foreground", foreground_frame)
        key = cv.waitKey(0)  # Waits for keypress
        if key == 27:  # Halt if ESC pressed
            exit()
        '''

        foreground_silhouettes.append(foreground_frame.copy()) # Append frame to array

        #print(f"background_subtraction: Processing frame {frame_count}")
        frame_count += 1

    vid.release()
    print(f"background_subtraction: Processed {frame_count} frames\n")
    return(foreground_silhouettes)

'''
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
'''