import numpy as np

class TrajectoryTracker:
    def __init__(self, trajectory_generator):
        self.traj = trajectory_generator
        self.last_s = 0.0  # start at beginning of path

    def find_nearest_s(self, pos, full_search=False, search_radius=1.0, resolution=100):
        """
        Find the trajectory parameter s that is closest to current position.
        Args:
            pos: np.array (3,) — current UAV position
            full_search: bool — if True, search entire spline
            search_radius: float — half-width of local search window
            resolution: int — number of samples to check
        Returns:
            s_closest: float — trajectory parameter that minimizes ||p - P̌(s)||
        """
        n = self.traj.n
        if full_search:
            s_values = np.linspace(0, n - 1, resolution * (n - 1))
        else:
            s_min = max(0, self.last_s - search_radius)
            s_max = min(n - 1, self.last_s + search_radius)
            s_values = np.linspace(s_min, s_max, resolution)

        # Evaluate distances
        dists = np.array([np.linalg.norm(self.traj.get_pos(s) - pos) for s in s_values])
        idx = np.argmin(dists)
        self.last_s = s_values[idx]  # update for next step
        return self.last_s