import numpy as np
import cv2
import math

class PathPlanning:
    def __init__(self, vehicle_width=2.0, vehicle_length=4.0):
        # Vehicle parameters
        self.vehicle_width = vehicle_width
        self.vehicle_length = vehicle_length

    def generate_path(self, start, goal, map_data):
        """
        Generate a basic path between the start and goal positions using A* algorithm.
        
        Parameters:
        start (tuple): Starting coordinates of the vehicle (x, y).
        goal (tuple): Goal coordinates (x, y).
        map_data (np.array): A map or occupancy grid to check for obstacles.
        
        Returns:
        path (list): List of (x, y) points representing the planned path.
        """
        # For simplicity, we are assuming we are using A* pathfinding here
        open_list = []
        closed_list = []
        g_costs = {}
        f_costs = {}
        came_from = {}
        
        open_list.append(start)
        g_costs[start] = 0
        f_costs[start] = self.heuristic(start, goal)

        while open_list:
            current = min(open_list, key=lambda x: f_costs[x])
            open_list.remove(current)
            closed_list.append(current)

            # If the goal is reached, reconstruct the path
            if current == goal:
                return self.reconstruct_path(came_from, current)

            neighbors = self.get_neighbors(current, map_data)
            for neighbor in neighbors:
                if neighbor in closed_list:
                    continue

                tentative_g_cost = g_costs[current] + self.distance(current, neighbor)
                if neighbor not in open_list:
                    open_list.append(neighbor)
                elif tentative_g_cost >= g_costs.get(neighbor, float('inf')):
                    continue

                came_from[neighbor] = current
                g_costs[neighbor] = tentative_g_cost
                f_costs[neighbor] = g_costs[neighbor] + self.heuristic(neighbor, goal)

        return []  # No valid path found

    def heuristic(self, current, goal):
        """
        Heuristic function (e.g., Euclidean distance) for A* search.
        
        Parameters:
        current (tuple): Current position (x, y).
        goal (tuple): Goal position (x, y).
        
        Returns:
        float: Estimated cost from current to goal.
        """
        return np.linalg.norm(np.array(current) - np.array(goal))

    def get_neighbors(self, point, map_data):
        """
        Get neighboring points around the current point considering obstacles.
        
        Parameters:
        point (tuple): Current position (x, y).
        map_data (np.array): A map or occupancy grid to check for obstacles.
        
        Returns:
        list: List of neighboring points (x, y).
        """
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

        for dx, dy in directions:
            nx, ny = point[0] + dx, point[1] + dy
            if 0 <= nx < map_data.shape[0] and 0 <= ny < map_data.shape[1] and map_data[nx, ny] == 0:
                neighbors.append((nx, ny))

        return neighbors

    def distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points.
        
        Parameters:
        point1 (tuple): First point (x, y).
        point2 (tuple): Second point (x, y).
        
        Returns:
        float: Distance between the two points.
        """
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def reconstruct_path(self, came_from, current):
        """
        Reconstruct the path from the goal to the start.
        
        Parameters:
        came_from (dict): A dictionary mapping nodes to their predecessors.
        current (tuple): The current node (goal).
        
        Returns:
        list: The planned path as a list of (x, y) points.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def visualize_path(self, img, path):
        """
        Visualize the planned path on the given image.
        
        Parameters:
        img (np.array): The input image (e.g., bird's-eye view of the environment).
        path (list): List of (x, y) points representing the planned path.
        
        Returns:
        np.array: Image with the planned path drawn.
        """
        for point in path:
            cv2.circle(img, tuple(point), 5, (0, 255, 0), -1)
        return img


# Example usage:
if __name__ == "__main__":
    # Create an instance of PathPlanning
    planner = PathPlanning()

    # Simulated map (0 represents free space, 1 represents an obstacle)
    map_data = np.zeros((100, 100))  # 100x100 grid of free space
    map_data[50:60, 50:60] = 1  # Adding an obstacle in the middle

    # Define start and goal positions
    start = (10, 10)
    goal = (90, 90)

    # Generate a path using A* algorithm
    path = planner.generate_path(start, goal, map_data)

    if path:
        print("Path found:", path)
    else:
        print("No path found!")

    # Visualize the path
    img = np.zeros((100, 100, 3), dtype=np.uint8)  # Create an empty image for visualization
    img_with_path = planner.visualize_path(img, path)

    # Display the image
    cv2.imshow('Planned Path', img_with_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
