import cv2
import numpy as np
import os
import yaml

# Define the chessboard dimensions (number of internal corners per row and column)
CHESSBOARD_ROWS = 6
CHESSBOARD_COLS = 9
CHESSBOARD_SIZE = (CHESSBOARD_COLS, CHESSBOARD_ROWS)

# Define the square size in the real world (e.g., in cm or meters, depending on your chessboard)
SQUARE_SIZE = 0.025  # Example: 25 mm square size

# Calibration file path
CALIB_PATH = "config/calibration.yaml"

def calibrate_stereo_cameras(left_frame, right_frame):
    """ Perform stereo calibration using chessboard corners. """
    
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...
    obj_points = np.zeros((CHESSBOARD_ROWS * CHESSBOARD_COLS, 3), np.float32)
    obj_points[:, :2] = np.indices((CHESSBOARD_COLS, CHESSBOARD_ROWS)).T.reshape(-1, 2)
    obj_points *= SQUARE_SIZE  # Scale the points based on square size
    
    # Arrays to store object points and image points from all frames
    object_points = []
    left_image_points = []
    right_image_points = []
    
    # Convert to grayscale
    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners in both left and right frames
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHESSBOARD_SIZE)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHESSBOARD_SIZE)

    if ret_left and ret_right:
        # Refine corner locations
        cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), 
                         cv2.TermCriteria(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), 
                         cv2.TermCriteria(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

        # Add object points and image points
        object_points.append(obj_points)
        left_image_points.append(corners_left)
        right_image_points.append(corners_right)

        # Draw the corners on the images (optional)
        cv2.drawChessboardCorners(left_frame, CHESSBOARD_SIZE, corners_left, ret_left)
        cv2.drawChessboardCorners(right_frame, CHESSBOARD_SIZE, corners_right, ret_right)
        
        cv2.imshow("Left Camera", left_frame)
        cv2.imshow("Right Camera", right_frame)
        cv2.waitKey(500)  # Wait for a brief moment to see the results

    # If we have at least one set of valid corners, we can perform stereo calibration
    if len(object_points) > 0:
        print("[INFO] Performing stereo calibration...")
        
        # Perform stereo camera calibration
        ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, E, F = cv2.stereoCalibrate(
            object_points, left_image_points, right_image_points, gray_left.shape[::-1], 
            None, None, None, None, 
            criteria=cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        # Calibration successful, return the parameters in a dictionary
        calib_data = {
            'camera_matrix_left': camera_matrix_left.tolist(),
            'camera_matrix_right': camera_matrix_right.tolist(),
            'dist_coeffs_left': dist_coeffs_left.tolist(),
            'dist_coeffs_right': dist_coeffs_right.tolist(),
            'R': R.tolist(),
            'T': T.tolist()
        }
        
        print("[INFO] Stereo calibration complete.")
        return calib_data
    else:
        print("[ERROR] Chessboard corners not detected.")
        return {}

def save_calibration(params):
    """ Save the calibration parameters to a YAML file. """
    with open(CALIB_PATH, 'w') as f:
        yaml.dump(params, f)
    print(f"[INFO] Calibration saved to {CALIB_PATH}")

def load_calibration():
    """ Load the calibration parameters from a YAML file. """
    if not os.path.exists(CALIB_PATH):
        print(f"[WARN] No calibration found at {CALIB_PATH}.")
        return None
    with open(CALIB_PATH, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def run_stereo_calibration(left_source, right_source):
    """ Run stereo calibration using video sources. """
    cam_left = cv2.VideoCapture(left_source)
    cam_right = cv2.VideoCapture(right_source)

    print("[INFO] Stereo Calibration started. Press 'c' to capture, 'q' to finish.")
    while True:
        ret_l, frame_l = cam_left.read()
        ret_r, frame_r = cam_right.read()
        if not ret_l or not ret_r:
            print("[ERROR] Could not read frames.")
            break

        display = np.hstack((frame_l, frame_r))
        cv2.imshow("Stereo Calibration", display)
        key = cv2.waitKey(1)
        if key == ord('c'):
            calib_data = calibrate_stereo_cameras(frame_l, frame_r)
            if calib_data:
                save_calibration(calib_data)
        elif key == ord('q'):
            break

    cam_left.release()
    cam_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example: python main.py --left left.mp4 --right right.mp4
    left_source = "/home/dexter/Projects/ADAS/data/test_video.mp4"  # Example left video file or camera index
    right_source = "/home/dexter/Projects/ADAS/data/test_video.mp4"  # Example right video file or camera index

    run_stereo_calibration(left_source, right_source)
