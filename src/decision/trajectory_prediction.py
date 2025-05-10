import numpy as np
import cv2
from filterpy.kalman import KalmanFilter

class TrajectoryPrediction:
    def __init__(self):
        # Kalman Filter Setup
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State vector (position and velocity in x, y)
        self.kf.x = np.array([0, 0, 0, 0])  # Initial state [x, y, vx, vy]
        
        # State transition matrix (predicts the next position based on current velocity)
        self.kf.F = np.array([[1, 0, 1, 0], 
                               [0, 1, 0, 1], 
                               [0, 0, 1, 0], 
                               [0, 0, 0, 1]])  # Transition matrix
        
        # Measurement function (maps the state space to the measurement space)
        self.kf.H = np.array([[1, 0, 0, 0], 
                               [0, 1, 0, 0]])  # We only measure positions
        
        # Measurement covariance matrix (uncertainty of measurements)
        self.kf.R = np.array([[0.1, 0], 
                               [0, 0.1]])  # We assume some measurement noise
        
        # Process noise covariance (uncertainty in process model)
        self.kf.Q = np.array([[0.1, 0, 0.1, 0], 
                               [0, 0.1, 0, 0.1], 
                               [0.1, 0, 0.1, 0], 
                               [0, 0.1, 0, 0.1]])  # Some process noise
        
        # Initial prediction
        self.kf.predict()

    def update(self, detection):
        """
        Update the Kalman Filter with the new detection.
        
        Parameters:
        detection (np.array): [x, y] position of the object at the current time step.
        
        Returns:
        np.array: Predicted future state [x, y, vx, vy] after the update.
        """
        # Update step with the new detection
        self.kf.update(detection)
        
        return self.kf.x

    def predict(self, steps=1):
        """
        Predict the future position of the object.
        
        Parameters:
        steps (int): Number of steps (time frames) to predict.
        
        Returns:
        np.array: Predicted future position [x, y].
        """
        predictions = []
        for _ in range(steps):
            self.kf.predict()  # Predict next step
            predictions.append(self.kf.x[:2])  # Only get [x, y] position
        
        return np.array(predictions)

    def visualize_trajectory(self, img, predicted_trajectory):
        """
        Visualize the predicted trajectory on the image.
        
        Parameters:
        img (np.array): The input image where the trajectory will be visualized.
        predicted_trajectory (np.array): Array of predicted [x, y] positions.
        
        Returns:
        img (np.array): Image with the predicted trajectory drawn.
        """
        for i in range(1, len(predicted_trajectory)):
            cv2.line(img, tuple(predicted_trajectory[i-1].astype(int)),
                     tuple(predicted_trajectory[i].astype(int)), (0, 255, 255), 2)
        
        return img


if __name__ == "__main__":
    # Example usage:
    tracker = TrajectoryPrediction()

    # Simulating object detections (replace with your object detection outputs)
    detections = np.array([[100, 150], [105, 155], [110, 160], [115, 165]])

    img = cv2.imread('test_frame.jpg')  # Load a frame from your video stream or stereo images

    predicted_trajectory = []

    for detection in detections:
        # Update the Kalman filter with each new detection
        state = tracker.update(detection)

        # Predict the trajectory after the detection
        future_positions = tracker.predict(steps=5)  # Predict next 5 steps

        predicted_trajectory.extend(future_positions)

    # Visualize the predicted trajectory on the image
    img_with_trajectory = tracker.visualize_trajectory(img, predicted_trajectory)

    # Show the result
    cv2.imshow('Predicted Trajectory', img_with_trajectory)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
