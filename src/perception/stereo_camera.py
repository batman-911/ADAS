import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from dsgn.models import StereoNet
from dsgn.models.inference3d import make_fcos3d_postprocessor
from tools.env_utils.exp import Experimenter

from src.utils.calibration import load_calibration

class DSGNSingleInference:
    def __init__(self, model_path, config_path=None, device='cuda'):
        """
        Initialize DSGN for single-image inference
        
        Args:
            model_path (str): Path to trained model weights (.tar)
            config_path (str): Optional path to config file
            device (str): 'cuda' or 'cpu'
        """
        self.device = device

        self.exp = Experimenter(os.path.dirname(model_path), config_path)
        self.cfg = self.exp.config
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load DSGN model with weights (without DataParallel)"""
        self.model = StereoNet(cfg=self.cfg).to(self.device)

        if model_path.endswith('.tar'):
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint['state_dict']

            # Supprime le préfixe "module." si présent
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # Supprime "module."
                else:
                    new_state_dict[k] = v

            self.model.load_state_dict(new_state_dict, strict=False)
            print(f'Loaded model from {model_path}')
        else:
            raise ValueError("Model file should be .tar archive")

        self.model.eval()


        
    def preprocess_images(self, left_img, right_img):
        """Convert images to float32 and normalize"""
        left = np.array(left_img).astype(np.float32) / 255.0  # Ensure float32
        right = np.array(right_img).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        left = (left - mean) / std
        right = (right - mean) / std
        
        left = torch.from_numpy(left).permute(2, 0, 1).unsqueeze(0).to(self.device)
        right = torch.from_numpy(right).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return left, right
    
    def prepare_calibration(self, calib_data):
        """Convert calibration matrices to float32"""
        calibs = {
            'fu': torch.tensor([calib_data['fu']], dtype=torch.float32, device=self.device),
            'baseline': torch.tensor([calib_data['baseline']], dtype=torch.float32, device=self.device),
            'P': torch.tensor(calib_data['P'], dtype=torch.float32, device=self.device),
            'P_R': torch.tensor(calib_data['P_R'], dtype=torch.float32, device=self.device)
        }
        return calibs
    
    def infer(self, left_img, right_img, calib_data):
        """
        Perform 3D object detection on stereo pair
        
        Args:
            left_img (PIL.Image): Left stereo image
            right_img (PIL.Image): Right stereo image
            calib_data (dict): Camera calibration data
            
        Returns:
            dict: Contains:
                - depth_map: Predicted depth map
                - detections: List of 3D bounding boxes (if RPN3D enabled)
        """
        # Preprocess inputs
        imgL, imgR = self.preprocess_images(left_img, right_img)
        calibs = self.prepare_calibration(calib_data)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(
                imgL, imgR, 
                calibs['fu'], calibs['baseline'],
                calibs['P'], calibs_Proj_R=calibs['P_R']
            )
        
        # Process outputs
        results = {
            'depth_map': outputs['depth_preds'][0].cpu().numpy()
        }
        
        # If 3D detection is enabled
        if hasattr(self.model.module, 'RPN3D_ENABLE') and self.model.module.RPN3D_ENABLE:
            image_size = (imgL.shape[2], imgL.shape[3])  # (H, W)
            box_pred = make_fcos3d_postprocessor(self.model.module.cfg)(
                outputs['bbox_cls'], outputs['bbox_reg'], 
                outputs['bbox_centerness'],
                image_sizes=[image_size],
                calibs_Proj=calibs['P']
            )
            
            # Convert detections to readable format
            results['detections'] = self.process_detections(box_pred[0])
        
        return results
    
    def process_detections(self, detections):
        """
        Convert model detections to readable format
        
        Args:
            detections (list): Raw detection output
            
        Returns:
            list: Processed detections with:
                - class_id
                - bbox_2d
                - bbox_3d
                - score
        """
        processed = []
        
        for det in detections:
            detection = {
                'class_id': det.get_field('labels').item(),
                'score': det.get_field('scores').item(),
                'bbox_2d': det.bbox.cpu().numpy().tolist()
            }
            
            if det.has_field('box_corner3d'):
                corners = det.get_field('box_corner3d').cpu().numpy()
                detection['bbox_3d'] = {
                    'corners': corners.reshape(8, 3).tolist(),
                    'center': corners.mean(axis=0).tolist()
                }
            
            processed.append(detection)
        
        return processed

# Example Usage
if __name__ == '__main__':
    # Initialize detector
    detector = DSGNSingleInference(
        model_path='/home/dexter/Projects/repo/DSGN/outputs/DSGN_car_12g/dsgn_12g_b/finetune_48.tar',
        config_path="/home/dexter/Projects/repo/DSGN/configs/config_car_12g.py",
        device='cpu'  # or 'cpu'
    )
    
    # Load stereo images
    left_img = Image.open('/home/dexter/Projects/kitti/data_scene_flow/training/image_2/000000_10.png').convert('RGB')
    right_img = Image.open('/home/dexter/Projects/kitti/data_scene_flow/training/image_3/000000_10.png').convert('RGB')
    
    # Prepare calibration data (example values - replace with real calibration)
    calib_data = {
        'fu': 721.5377,  # Focal length
        'baseline': 0.54,  # Stereo baseline
        'P': np.array([[721.5377, 0, 609.5593, 44.85728],
                       [0, 721.5377, 172.854, 0],
                       [0, 0, 1, 0]]),  # Left camera projection matrix
        'P_R': np.array([[721.5377, 0, 609.5593, -44.85728],
                         [0, 721.5377, 172.854, 0],
                         [0, 0, 1, 0]])  # Right camera projection matrix
    }
    
    # Run inference
    results = detector.infer(left_img, right_img, calib_data)
    
    # Display results
    print("Depth map shape:", results['depth_map'].shape)
    if 'detections' in results:
        print(f"Found {len(results['detections'])} objects:")
        for i, det in enumerate(results['detections']):
            print(f"Object {i+1}:")
            print(f"  Class: {'Car' if det['class_id'] == 2 else 'Pedestrian' if det['class_id'] == 1 else 'Cyclist'}")
            print(f"  Confidence: {det['score']:.2f}")
            print(f"  2D BBox: {det['bbox_2d']}")
            if 'bbox_3d' in det:
                print(f"  3D Center: {det['bbox_3d']['center']}")

if __name__ == '__main__':

    # Load the stereo calibration
    calib_data = load_calibration()

    # Initialize the depth estimator
    stereo_camera = StereoCameraModel()

    # Define the input images and output directory
    left_imgs_glob = "/home/dexter/Projects/ADAS-Project/assets/image_2/000000_10.png"
    right_imgs_glob = "/home/dexter/Projects/ADAS-Project/assets/image_3/000000_10.png"

    # Process images and estimate depth
    disparity_map = depth_estimator.estimate_depth(left_imgs_glob, right_imgs_glob)
    depth_estimator.save_disp(disparity_map)


