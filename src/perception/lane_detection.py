import cv2
import numpy as np
import torch
from torchvision import transforms
from model import LaneNet  # Assuming you have a LaneNet model in 'model.py' or use SCNN

class LaneDetection:
    def __init__(self, model_path):
        # Initialize the model and load the pre-trained weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LaneNet().to(self.device)  # Load the model (LaneNet or SCNN)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set the model to evaluation mode

    def preprocess_image(self, img):
        """
        Preprocess the input image to fit the model's requirements
        - Resize
        - Normalize
        - Convert to tensor
        """
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = transform(img)
        return img.unsqueeze(0).to(self.device)  # Add batch dimension and move to device

    def detect_lane(self, img):
        """
        Detect lanes on the input image
        """
        # Preprocess image
        img_tensor = self.preprocess_image(img)
        
        with torch.no_grad():
            # Run the image through the lane detection model
            lane_map = self.model(img_tensor)
        
        return lane_map.cpu().numpy()  # Convert the result back to a NumPy array

    def visualize_lanes(self, img, lane_map):
        """
        Visualize the detected lanes on the original image
        """
        lane_map = lane_map.squeeze()  # Remove batch dimension
        lane_map = np.where(lane_map > 0.5, 1, 0)  # Apply threshold to the lane map

        # Overlay the lane map onto the original image
        lane_img = img.copy()
        lane_img[lane_map == 1] = [0, 0, 255]  # Red color for lanes (BGR format)
        
        return lane_img

    def run(self, img):
        """
        Full lane detection pipeline
        """
        # Detect lanes
        lane_map = self.detect_lane(img)

        # Visualize lanes on the image
        lane_img = self.visualize_lanes(img, lane_map)

        return lane_img


if __name__ == "__main__":
    # Example usage
    lane_detector = LaneDetection(model_path="path_to_model_weights.pth")
    
    # Read a stereo image (left or right image from the stereo pair)
    img = cv2.imread('/home/dexter/Projects/ADAS/data/image_2/000000_10.png')
    
    # Run lane detection
    lane_img = lane_detector.run(img)
    
    # Show the result
    cv2.imshow('Lane Detection', lane_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
