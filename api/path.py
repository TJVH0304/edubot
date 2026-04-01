'''
Custom classes for generating paths. The possible paths are:

- RobotPath: simply providing points
- TrianglePath: a triangle in the XZ plane
- FigureEightPath: an infinity symbol in the XZ plane
- PickAndPlacePath: the pick and place trajectory as outlined in the assignment
- StackingPath: the stacking block path as outlined in the assignment
'''

import numpy as np


class RobotPath:
    def __init__(self, points):
        self.points = points


class TrianglePath(RobotPath):
    def __init__(self, corner_points, points_per_edge=10, closed=True):
        corner_points = np.asarray(corner_points, dtype=np.float64)

        if corner_points.shape != (3, 3):
            raise ValueError('corner points needa be 3x3')

        a, b, c = corner_points

        edge_ab = self._interpolate(a, b, points_per_edge, include_endpoint=False)
        edge_bc = self._interpolate(b, c, points_per_edge, include_endpoint=False)
        edge_ca = self._interpolate(c, a, points_per_edge, include_endpoint=closed)

        path_points = np.vstack((edge_ab, edge_bc, edge_ca))

        super().__init__(path_points)

    @staticmethod
    def _interpolate(p1, p2, points_per_edge, include_endpoint=False):
        t = np.linspace(0, 1, points_per_edge, endpoint=include_endpoint)
        return p1[None, :] + (p2 - p1)[None, :] * t[:, None]

class FigureEightPath(RobotPath):
    def __init__(self, center=(0, 0, 0), scale=0.3, points=100, closed=True):
        center = np.asarray(center, dtype=np.float64)

        if center.shape != (3,):
            raise ValueError("center must be a 3D point")

        t = np.linspace(0, 2 * np.pi, points, endpoint=closed)

        # Figure-eight in XZ plane (y fixed)
        x = np.sin(t)
        y = np.zeros_like(t) + center[1]
        z = np.sin(t) * np.cos(t)

        path = np.stack((x, y, z), axis=1)

        # Scale and translate
        path = scale * path + center[None, :]

        super().__init__(path)

class PickAndPlacePath(RobotPath):
    def __init__(self, load_point, target_point, home_point, h_blocks=0.02, points_per_edge=2, closed=True):
        self.load_point = load_point
        self.target_point = target_point
        self.home_point = home_point
        self.clearance_z = h_blocks
        self.points_per_edge = points_per_edge
        self.closed = closed

        self.__pickup_idx = []
        self.__place_idx = []

        points = self._interpolate()

        super().__init__(points)

    def _interpolate(self):
        load = np.asarray(self.load_point, dtype=float)
        target = np.asarray(self.target_point, dtype=float)
        home = np.asarray(self.home_point, dtype=float)

        def segment(p1, p2):
            t = np.linspace(0, 1, self.points_per_edge)
            return p1[None, :] + (p2 - p1)[None, :] * t[:, None]

        def lift(p):
            p_up = p.copy()
            p_up[2] = self.clearance_z + self.load_point[-1]
            return p_up

        path = []

        def move(p1, p2):
            path.append(segment(p1, p2))

        def waypoint(p):
            path.append(p[None, :])

        # Precompute lifted points
        load_up = lift(load)
        target_up = lift(target)

        # ---- FORWARD: load → target ----
        move(home, load_up)
        move(load_up, load)
        waypoint(load)          # PICK
        self.__pickup_idx.append(len(np.vstack(path)) - 1)

        move(load, load_up)
        move(load_up, target_up)
        move(target_up, target)
        waypoint(target)        # PLACE
        self.__place_idx.append(len(np.vstack(path)) - 1)

        move(target, target_up)
        move(target_up, home)

        # ---- BACKWARD: target → load ----
        move(home, target_up)
        move(target_up, target)
        waypoint(target)        # PICK
        self.__pickup_idx.append(len(np.vstack(path)) - 1)

        move(target, target_up)
        move(target_up, load_up)
        move(load_up, load)
        waypoint(load)          # PLACE
        self.__place_idx.append(len(np.vstack(path)) - 1)

        move(load, load_up)
        move(load_up, home)

        return np.vstack(path)

    @property
    def pickup_points(self):
        return self.points[self.__pickup_idx]

    @property
    def place_points(self):
        return self.points[self.__place_idx]
    
    @property
    def pick_idx(self):
        return self.__pickup_idx
    
    @property
    def place_idx(self):
        return self.__place_idx
    
class StackingPath(RobotPath):
    def __init__(self, load_point, place_point, z_clearance=0.02, h_blocks=0.02, n_blocks=3, points_per_edge=2, closed=True):
        self.load_point = load_point
        self.target_point = place_point
        self.clearance_z = z_clearance
        self.h_blocks = h_blocks
        self.n_blocks = n_blocks
        self.points_per_edge = points_per_edge
        self.closed = closed

        self.__place_idx = []
        self.__pickup_idx = []
        
        points = self._interpolate()

        super().__init__(points)

    def _interpolate(self):
        load = np.asarray(self.load_point, dtype=float)
        target0 = np.asarray(self.target_point, dtype=float)

        def segment(p1, p2):
            t = np.linspace(0, 1, self.points_per_edge)
            return p1[None, :] + (p2 - p1)[None, :] * t[:, None]

        def lift(p, n_block=0):
            p_up = p.copy()
            p_up[2] = self.clearance_z + self.load_point[-1] + n_block * self.h_blocks
            return p_up

        def target_n(n):
            tar = target0.copy()
            tar[-1] += n * self.h_blocks
            return tar

        path = []

        def move(p1, p2):
            path.append(segment(p1, p2))

        def waypoint(p):
            path.append(p[None, :])

        # Precompute lifted points
        
        
        
        # TODO: make for loop, and use self.n_blocks
        # pick up block 1
        for i in range(self.n_blocks):
            load_up = lift(load, i)
            move(load_up, load)
            waypoint(load)
            self.__pickup_idx.append(len(np.vstack(path)) - 1)

            move(load, load_up)

            target = target_n(i)
            target_up = lift(target, i)
            move(load_up, target_up)

            move(target_up, target)
            waypoint(target)
            self.__place_idx.append(len(np.vstack(path)) - 1)
            move(target, target_up)

            if i != self.n_blocks - 1:
                move(target_up, load_up)

        # move(load_up, load)
        # waypoint(load) # pick
        # self.__pickup_idx.append(len(np.vstack(path)) - 1)

        # move(load, load_up)

        # # move to target area
        # target = target_n(0)
        # target_up = lift(target, 0)
        # move(load_up, target_up)

        # # place block 1
        # move(target_up, target)
        # waypoint(target) # place
        # self.__place_idx.append(len(np.vstack(path)) - 1)
        # move(target, target_up)

        # # move to block 2 load
        # move(target_up, load_up)

        # # pick up block 2
        # move(load_up, load)
        # waypoint(load) # pick
        # self.__pickup_idx.append(len(np.vstack(path)) - 1)

        # move(load, load_up)

        # # move to target area
        # target = target_n(1)
        # target_up = lift(target, 1)
        # move(load_up, target_up)

        # # place block 2
        # move(target_up, target)
        # waypoint(target) # place
        # self.__place_idx.append(len(np.vstack(path)) - 1)
        # move(target, target_up)

        return np.vstack(path)



    @property
    def pickup_points(self):
        return self.points[self.__pickup_idx]

    @property
    def place_points(self):
        return self.points[self.__place_idx]
    
    @property
    def pick_idx(self):
        return self.__pickup_idx
    
    @property
    def place_idx(self):
        return self.__place_idx


    