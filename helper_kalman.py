import numpy as np
from scipy.linalg import inv


class KalmanFilter:
    """
    Kalman filter for 2D position tracking with constant velocity model.
    State vector: [x, y, vx, vy] (position and velocity)
    """
    
    def __init__(self, initial_x=0, initial_y=0, process_noise=0.1, measurement_noise=1.0):
        # State vector: [x, y, vx, vy]
        self.state = np.array([initial_x, initial_y, 0, 0], dtype=float)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        # Error covariance matrix
        self.P = np.eye(4) * 1000  # Large initial uncertainty
        
        self.initialized = False
    
    def predict(self):
        """Predict next state"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement):
        """Update state with measurement"""
        if not self.initialized:
            # Initialize with first measurement
            self.state[0] = measurement[0]
            self.state[1] = measurement[1]
            self.initialized = True
            return
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ inv(S)
        
        # Update state
        y = measurement - (self.H @ self.state)
        self.state = self.state + K @ y
        
        # Update error covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
    
    def get_position(self):
        """Get current position estimate"""
        return self.state[0], self.state[1]
    
    def get_velocity(self):
        """Get current velocity estimate"""
        return self.state[2], self.state[3]


def smooth_trajectory_with_kalman(centroids_per_vehicle_id, process_noise=0.1, measurement_noise=1.0):
    """
    Apply Kalman filter to smooth vehicle trajectories.
    
    Args:
        centroids_per_vehicle_id: Dictionary with vehicle IDs as keys and lists of 
                                 centroid points as values
        process_noise: Process noise parameter (higher = more responsive to changes)
        measurement_noise: Measurement noise parameter (higher = more trust in model)
    
    Returns:
        Dictionary with smoothed trajectories per vehicle
    """
    smoothed_trajectories = {}
    
    for vehicle_id, centroids in centroids_per_vehicle_id.items():
        if len(centroids) < 2:
            # Not enough points to smooth
            smoothed_trajectories[vehicle_id] = centroids
            continue
        
        # Initialize Kalman filter with first point
        first_point = centroids[0]
        kf = KalmanFilter(
            initial_x=first_point['centroid_x'],
            initial_y=first_point['centroid_y'],
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )
        
        smoothed_centroids = []
        
        for i, point in enumerate(centroids):
            # Predict next state
            kf.predict()
            
            # Update with measurement
            measurement = np.array([point['centroid_x'], point['centroid_y']])
            kf.update(measurement)
            
            # Get smoothed position
            smoothed_x, smoothed_y = kf.get_position()
            
            # Create smoothed point with same structure as original
            smoothed_point = {
                'frame_number': point['frame_number'],
                'centroid_x': smoothed_x,
                'centroid_y': smoothed_y
            }
            smoothed_centroids.append(smoothed_point)
        
        smoothed_trajectories[vehicle_id] = smoothed_centroids
    
    return smoothed_trajectories


def apply_kalman_smoothing(centroids_per_vehicle_id, process_noise=0.1, measurement_noise=1.0):
    smoothed = smooth_trajectory_with_kalman(centroids_per_vehicle_id, process_noise, measurement_noise)
    total_points_before = sum(len(centroids) for centroids in centroids_per_vehicle_id.values())
    total_points_after = sum(len(centroids) for centroids in smoothed.values())
    
    return smoothed, total_points_before, total_points_after
