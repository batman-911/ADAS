import cv2
from models.lane_detection_model import CLRNetDemo

class LaneDetector:
    def __init__(self, config):
        """
        Initialize the LaneDetection model with parameters from config.
        """
        self.model_params = config.get("perception", "lane_detection")
        self.model = CLRNetDemo(**self.model_params)
    
    def detect_lanes(self, img):
        """
        Detect lanes in the input image.
        """
        output = self.model.forward(img)
        return output 

    def visualize_lanes(self, img, output, thickness=2):
        """
        Visualize detected lanes on the image.
        """
        res = self.model.imshow_lanes(img, output[0], width=thickness)
        cv2.imshow("Lanes Visualization", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    from config.config import Config
    
    # Load configuration
    config = Config()
    
    # Read an image (e.g., left or right image from the stereo pair)
    img = cv2.imread('data/image_2/000000_10.png')
    
    # Initialize LaneDetection class with configuration
    lane_detector = LaneDetector(config)

    # Run lane detection
    output = lane_detector.detect_lanes(img)
    print(output)
    
    # Visualize the detected lanes
    lane_detector.visualize_lanes(img, output[0])
