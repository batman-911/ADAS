import torch
import numpy as np

from src.utils.calibration import load_calibration

class StereoCameraModel:
    def __init__(self, model_config):

        self.device = model_config['device']
        self.model_config = model_config['perception']['stereo_camera']
        
        if self.device == "cuda":
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            self.model = torch.nn.DataParallel(IGEVStereo(self.args), device_ids=[0])
            self.model.load_state_dict(torch.load(self.args.restore_ckpt))

            self.model = self.model.module
        
        # cpu mode 
        self.model = IGEVStereo(self.args)
        checkpoint = torch.load(self.args.restore_ckpt, map_location=self.device)
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

    def _preprocess_image(self, frame):
        pass

    def estimate_depth(self, left_frame, right_frame):
        """
        Estimate depth from a pair of stereo images.
        """
        left_frame = self._preprocess_image(left_frame)
        right_frame = self._preprocess_image(right_frame)


        # Prepare the images for model input
        padder = InputPadder(left_img.shape, divis_by=32)
        left_img, right_img = padder.pad(left_img, right_img)

        # Perform depth estimation
        with torch.no_grad():
            disp = self.model(left_img, right_img, test_mode=True)
            disp = padder.unpad(disp)
        
        return disp

    def save_disp(self, disp):
        """
        Save the output disp map to the specified directory.
        """
        output_directory = Path(self.args.output_directory)
        output_directory.mkdir(exist_ok=True)

        disp = disp.cpu().numpy().squeeze()
        plt.imsave(output_directory / '0.png', disp, cmap='jet')

        if self.args.save_numpy:
            np.save(output_directory / "0.npy", disp)

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


