import sys
import os
# Get the root directory dynamically (2 levels up from this file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)

import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from models.lane_detection_model import PolyRegression


class LaneDetection:
    def __init__(self, config):
        self.config = config
        self.device = config.get_device()
        self.model_params = self.config.get("perception", "lane_detection")

        self.transform = transforms.Compose([
            transforms.Resize((360, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = PolyRegression(**self.model_params["params"]).to(self.device)  
        self.model.load_state_dict(torch.load(self.model_params["model_path"], map_location=self.device)['model'])
        self.model.eval()  # Set the model to evaluation mode

    def _preprocess_image(self, img):
        """
        Preprocess the input image to fit the model's requirements
        - Resize
        - Normalize
        - Convert to tensor
        """
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        return self.transform(pil_image).unsqueeze(0).to(self.device)  
    
    def detect_lane(self, img):
        """
        Detect lanes on the input image
        """
        # Preprocess image
        img_tensor = self._preprocess_image(img)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            lane_output, _ = self.model.decode(outputs, labels=None, conf_threshold=self.model_params["conf_threshold"])
        
        return lane_output.cpu().numpy()  # Convert the result back to a NumPy array

    def visualize_lanes(self, img, lanes, color=(0, 0, 255), thickness=2, blend_weight=0.6):
        """
        Draw detected lanes on the input image.

        Args:
            img (np.ndarray): Original BGR image.
            lanes (np.ndarray): (N, 7) array where each row represents a lane:
                                [conf, y_lower, y_upper, poly3, poly2, poly1, poly0]
            color (tuple): Color of the lane lines (default red).
            thickness (int): Thickness of the lane lines.
            blend_weight (float): Blending weight for overlay.
        """
        img_h, img_w = img.shape[:2]
        overlay = img.copy()

        # Filter out invalid lanes
        valid_lanes = lanes[lanes[:, 0] > 0]

        for lane in valid_lanes:

            lane = lane[1:]  # remove conf
            lower, upper = lane[0], lane[1]
            lane = lane[2:]  # remove upper, lower positions

            # generate points from the polynomial
            ys = np.linspace(lower, upper, num=100)
            points = np.zeros((len(ys), 2), dtype=np.int32)
            points[:, 1] = (ys * img_h).astype(int)
            points[:, 0] = (np.polyval(lane, ys) * img_w).astype(int)
            points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]

            # draw lane with a polyline on the overlay
            # draw lane with a polyline on the overlay
            for current_point, next_point in zip(points[:-1], points[1:]):
                overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=color, thickness=2)

        w = 0.6
        img = ((1. - w) * img + w * overlay).astype(np.uint8)
        cv2.imshow("Lanes Visualization", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    from config.config import Config
    
    config = Config()
    # Read a stereo image (left or right image from the stereo pair)
    img = cv2.imread('/home/dexter/Projects/ADAS/data/image_2/000007_10.png')

    lane_detector = LaneDetection(config)

    # Run lane detection
    lane_output = lane_detector.detect_lane(img)
    print(lane_output)
    
    # visualization
    lane_detector.visualize_lanes(img, lane_output[0])


