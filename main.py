import argparse
import cv2


from src.perception.stereo_camera import StereoCameraModel
from src.perception.lane_detection import LaneDetector
from src.perception.object_detection import ObjectDetector3D
from src.perception.object_tracking import ObjectTracker
from src.decision.trajectory_prediction import TrajectoryPredictor
from src.decision.path_planning import PathPlanner
from src.utils.visualizer import show_result
from src.utils.logger import log_decision
from src.utils.calibration import run_stereo_calibration
from config.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Level 3 ADAS CLI")
    parser.add_argument('--left', type=str, help="Path to left video")
    parser.add_argument('--right', type=str, help="Path to right video")
    parser.add_argument('--calibrate', action='store_true', help="Run stereo calibration")
    return parser.parse_args()

class ADAS:
    def __init__(self, args):
        self.config = Config()

        self.stereo_camera = StereoCameraModel(self.config)
        self.lane_detector = LaneDetector(self.config)
        self.object_detector = ObjectDetector3D(self.config)
        self.object_tracker = ObjectTracker(self.config)
        self.trajectory_predictor = TrajectoryPredictor(self.config)
        self.path_planner = PathPlanner(self.config)

        self.left_cap = cv2.VideoCapture(args.left)
        self.right_cap = cv2.VideoCapture(args.right)


    def run(self):
        while True:
            ret_left, left = self.left_cap.read()
            ret_right, right = self.right_cap.read()
            if not ret_left or not ret_right:
                break

            depth_map = self.stereo_camera.estimate_depth(left, right)
            
            lanes_2d = self.lane_detector.detect(left)
            lanes_3d = self.stereo_camera.lift_to_3d(lanes_2d, depth_map)

            objects_2d = self.object_detector.detect(left)
            objects_3d = self.stereo_camera.estimate_3d_objects(objects_2d, depth_map)

            tracked = self.object_tracker.track(objects_3d)
            predicted = self.trajectory_predictor.predict(tracked)
            path = self.path_planner.plan(lanes_3d, predicted)

            log_decision(path)
            show_result(left, lanes_2d, objects_2d, path)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.left_cap.release()
        self.right_cap.release()
        cv2.destroyAllWindows()

def main():
    args = parse_args()
    if args.calibrate:
        run_stereo_calibration(args.left, args.right)

    adas = ADAS(args)
    adas.run()

if __name__ == "__main__":
    main()
