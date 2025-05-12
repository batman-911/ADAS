import numpy as np
import math

class VehicleController:
    def __init__(self, max_steering_angle, max_throttle, max_brake):
        self.max_steering_angle = max_steering_angle  # Maximum steering angle in radians
        self.max_throttle = max_throttle  # Maximum throttle value (usually from 0 to 1)
        self.max_brake = max_brake  # Maximum brake force (usually from 0 to 1)
        
    def compute_steering(self, lane_center, vehicle_position, lane_width):
        """
        Compute the steering angle based on the position of the vehicle relative to the lane.
        
        Args:
            lane_center (tuple): (x, y) coordinates of the lane center.
            vehicle_position (tuple): (x, y) coordinates of the vehicle.
            lane_width (float): The width of the lane.
        
        Returns:
            float: Steering angle command.
        """
        # Calculate the vehicle's lateral error from the lane center
        error = lane_center[0] - vehicle_position[0]
        
        # Proportional controller for steering (simplified)
        steering_angle = np.clip(error / lane_width, -self.max_steering_angle, self.max_steering_angle)
        
        return steering_angle

    def compute_throttle(self, speed, target_speed):
        """
        Compute the throttle value to reach the target speed.
        
        Args:
            speed (float): Current speed of the vehicle.
            target_speed (float): Desired speed.
        
        Returns:
            float: Throttle value (0 to 1).
        """
        speed_error = target_speed - speed
        
        # Simple proportional controller for speed control
        throttle = np.clip(speed_error / target_speed, 0, self.max_throttle)
        
        return throttle

    def compute_brake(self, speed, min_safe_speed):
        """
        Compute the brake value when the vehicle needs to slow down.
        
        Args:
            speed (float): Current speed of the vehicle.
            min_safe_speed (float): The minimum safe speed (e.g., when obstacles are too close).
        
        Returns:
            float: Brake value (0 to 1).
        """
        if speed < min_safe_speed:
            return 0  # No brake needed if we're already below the safe speed.
        
        # Simple braking based on the difference from the safe speed
        brake_force = np.clip((speed - min_safe_speed) / speed, 0, self.max_brake)
        
        return brake_force

    def control_vehicle(self, lane_center, vehicle_position, lane_width, speed, target_speed, min_safe_speed):
        """
        Compute the necessary control commands (steering, throttle, brake).
        
        Args:
            lane_center (tuple): (x, y) of lane center for steering.
            vehicle_position (tuple): (x, y) position of the vehicle.
            lane_width (float): Width of the lane for steering correction.
            speed (float): Current speed of the vehicle.
            target_speed (float): Target speed to be reached.
            min_safe_speed (float): Minimum safe speed for braking.
        
        Returns:
            dict: Steering, throttle, and brake commands.
        """
        steering = self.compute_steering(lane_center, vehicle_position, lane_width)
        throttle = self.compute_throttle(speed, target_speed)
        brake = self.compute_brake(speed, min_safe_speed)
        
        return {
            'steering': steering,
            'throttle': throttle,
            'brake': brake
        }
