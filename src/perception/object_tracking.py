import cv2
import numpy as np
from sort import Sort  # You need to install the SORT algorithm separately (https://github.com/abewley/sort)

class ObjectTracking:
    def __init__(self):
        # Initialize SORT tracker
        self.tracker = Sort()

    def track_objects(self, detections):
        """
        Perform tracking on detected objects.
        
        Parameters:
        detections (numpy.array): Array of detections in the format [x1, y1, x2, y2, score]
                                  (x1, y1) - top-left corner of bounding box
                                  (x2, y2) - bottom-right corner of bounding box
                                  (score) - confidence score of detection
        
        Returns:
        numpy.array: Array of tracked objects, each with [x1, y1, x2, y2, object_id]
        """
        # Use the SORT tracker to update object locations and maintain object IDs
        tracked_objects = self.tracker.update(detections)
        return tracked_objects

    def visualize_tracking(self, img, tracked_objects):
        """
        Visualize tracked objects by drawing bounding boxes and object IDs.
        
        Parameters:
        img (numpy.array): Input image on which to overlay the bounding boxes.
        tracked_objects (numpy.array): Array of tracked objects, each with [x1, y1, x2, y2, object_id]
        
        Returns:
        img (numpy.array): Image with bounding boxes and object IDs drawn.
        """
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            # Draw bounding box and object ID
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green color
            cv2.putText(img, f"ID: {int(obj_id)}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return img

    def run(self, img, detections):
        """
        Full object tracking pipeline.
        
        Parameters:
        img (numpy.array): Input image to track objects on.
        detections (numpy.array): Array of object detections for this frame.
        
        Returns:
        img (numpy.array): Image with tracked objects and their IDs.
        """
        # Perform object tracking
        tracked_objects = self.track_objects(detections)

        # Visualize tracked objects
        img = self.visualize_tracking(img, tracked_objects)

        return img


if __name__ == "__main__":
    # Initialize object tracking
    tracker = ObjectTracking()
    
    # Example usage with dummy detections (replace with your object detection model's output)
    # Example format of detections: [x1, y1, x2, y2, score]
    # For simplicity, we're simulating a single object detected at [100, 150, 200, 250] with a confidence score of 0.9
    detections = np.array([[100, 150, 200, 250, 0.9]])

    # Read a frame (this can be a video frame or a stereo image frame)
    img = cv2.imread('test_frame.jpg')

    # Run object tracking
    tracked_img = tracker.run(img, detections)

    # Show the result
    cv2.imshow('Tracked Objects', tracked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
