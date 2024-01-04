"""
This program can be launched directly.
Example of how to use semantic sensor, grasping and dropping
"""

import time
import os
import sys
import random
import math
from typing import Optional, List, Type
from enum import Enum

import numpy as np
from spg.utils.definitions import CollisionTypes

import networkx as nx
from scipy.spatial.distance import euclidean
from heapq import heappop, heappush

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.entities.rescue_center import RescueCenter, wounded_rescue_center_collision
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle, circular_mean
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.grid import Grid
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR

import cv2


class OccupancyGrid(Grid):
    """Simple occupancy grid"""

    def __init__(self,
                 size_area_world,
                 resolution: float,
                 lidar):
        super().__init__(size_area_world=size_area_world, resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.lidar = lidar

        self.x_max_grid: int = int(self.size_area_world[0] / self.resolution + 0.5)
        self.y_max_grid: int = int(self.size_area_world[1] / self.resolution + 0.5)

        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))
        self.zoomed_grid = np.empty((self.x_max_grid, self.y_max_grid))

    def update_grid(self, pose: Pose, reallyUpdate = True):
        """
        Bayesian map update with new observation
        lidar : lidar data
        pose : corrected pose in world coordinates
        """
        if reallyUpdate:
            EVERY_N = 3
            LIDAR_DIST_CLIP = 40.0
            EMPTY_ZONE_VALUE = -0.602
            OBSTACLE_ZONE_VALUE = 2.0
            FREE_ZONE_VALUE = -4.0
            THRESHOLD_MIN = -60
            THRESHOLD_MAX = 40

            lidar_dist = self.lidar.get_sensor_values()[::EVERY_N].copy()
            lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()

            # Compute cos and sin of the absolute angle of the lidar
            cos_rays = np.cos(lidar_angles + pose.orientation)
            sin_rays = np.sin(lidar_angles + pose.orientation)

            max_range = MAX_RANGE_LIDAR_SENSOR * 0.9

            # For empty zones
            # points_x and point_y contains the border of detected empty zone
            # We use a value a little bit less than LIDAR_DIST_CLIP because of the noise in lidar
            lidar_dist_empty = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
            # All values of lidar_dist_empty_clip are now <= max_range
            lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_range)
            points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip, cos_rays)
            points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip, sin_rays)

            for pt_x, pt_y in zip(points_x, points_y):
                self.add_value_along_line(pose.position[0], pose.position[1], pt_x, pt_y, EMPTY_ZONE_VALUE)

            # For obstacle zones, all values of lidar_dist are < max_range
            select_collision = lidar_dist < max_range

            points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
            points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)

            points_x = points_x[select_collision]
            points_y = points_y[select_collision]

            self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)

            # the current position of the drone is free !
            self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

            # threshold values
            self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

        # compute zoomed grid for displaying
        self.zoomed_grid = self.grid.copy()
        new_zoomed_size = (int(self.size_area_world[1] * 0.5), int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size, interpolation=cv2.INTER_NEAREST)


counter = 0
path = []

class MyDroneExploreMix(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4
        SLAVE = 5

        

    def __init__(self,
                 identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         display_lidar_graph=False,
                         **kwargs)
        # The state is initialized to searching wounded person
        if self.identifier == 0:
            self.state = self.Activity.SEARCHING_WOUNDED
        else:
            self.state = self.Activity.SLAVE           
        # Those values are used by the random control function
        self.counterStraight = 0
        self.angleStopTurning = 0
        self.isTurning = False
        self.tryingToFace = False

        self.goingBackwards = False

        self.iteration: int = 0

        self.estimated_pose = Pose()

        resolution = 8
        self.occupancyGrid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())

        self.bonusOccupancyGrid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())
    
        self.binaryGrid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())

        self.binaryClone = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())

        self.astarPathFound = False
        self.astarPath = []
        self.explorePath = []
        self.bonusOccupancyGrid.update_grid(pose=self.estimated_pose)
        

    def define_message_for_all(self):
        if self.state is self.Activity.SLAVE:
            return None
        else:
            msg_data = (self.identifier,
                        (self.measured_gps_position(), self.measured_compass_angle()))
            return msg_data

        pass

    def control(self):
        global counter, path
        if counter == 0:
            self.occupancyGrid.update_grid(pose=self.estimated_pose)
            self.bonusOccupancyGrid.grid = self.occupancyGrid.grid.copy()
        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()), self.measured_compass_angle())
        #time.sleep(0.1)
        self.occupancyGrid.update_grid(pose=self.estimated_pose)
        self.binaryGrid.update_grid(pose=self.estimated_pose)
        self.binaryClone.update_grid(pose=self.estimated_pose)
        
        if counter % 5 == 0:
            self.occupancyGrid.display(self.occupancyGrid.zoomed_grid, self.estimated_pose, title="zoomed occupancy grid")
            #self.occupancyGrid.display(self.occupancyGrid.grid, self.estimated_pose, title="occupancy grid")
            #self.occupancyGrid.display(self.occupancyGrid.zoomed_grid, self.estimated_pose, title="zoomed occupancy grid")
            
        counter += 1
        #print(counter)
        #print(self.estimated_pose.position)
        #print(self.estimated_pose.orientation)

        self.occupancyGrid.update_grid(pose=self.estimated_pose)

        
        self.bonusOccupancyGrid.grid = self.occupancyGrid.grid.copy()
        rows, cols = self.bonusOccupancyGrid.grid.shape
        for i in range(rows):
            for j in range(cols):
                if self.occupancyGrid.grid[i, j] > 20:
                    self.bonusOccupancyGrid.grid[i, j] = 40
                

        self.bonusOccupancyGrid.grid[self.bonusOccupancyGrid.grid > 20] = 40

        self.bonusOccupancyGrid.display(self.bonusOccupancyGrid.zoomed_grid, self.estimated_pose, title="zoomed bonusOccupancy grid")

        self.bonusOccupancyGrid.update_grid(pose=self.estimated_pose, reallyUpdate= False)
        

        # Iterate over the grid to find squares within 4 squares of a 40
        for i in range(self.bonusOccupancyGrid.grid.shape[0]):
            for j in range(self.bonusOccupancyGrid.grid.shape[1]):
                if self.bonusOccupancyGrid.grid[i, j] == 40:
                    # Set squares within 4 squares of a 40 to 30
                    for x in range(max(0, i - 4), min(self.bonusOccupancyGrid.grid.shape[0], i + 5)):
                        for y in range(max(0, j - 4), min(self.bonusOccupancyGrid.grid.shape[1], j + 5)):
                            self.bonusOccupancyGrid.grid[x, y] = 30

        #print(self.occupancyGrid.grid[0][0])
        #print(self.bonusOccupancyGrid.grid[0][0])

        self.getImbalanceList()
        if max(self.imbalanceList) > 30:
            components = calculate_movement_components(self.imbalanceList.index(min(self.imbalanceList)))
            if self.imbalanceList.index(min(self.imbalanceList)) < 90:
                turningComponent = -1
            else : 
                turningComponent = 1
        else : 
            components = 0,0
            turningComponent = 0


        command = {"forward": 0.0, #1 = forward
                   "lateral": 0.0,
                   "rotation": 0.0, #1 = left
                   "grasper": 0}

        found_wounded, found_rescue_center, command_semantic = self.process_semantic_sensor()

        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

        if self.state is self.Activity.SLAVE :
            found_drone, command_comm = self.process_communication_sensor()
            if found_drone:
                command = command_comm
                command["grasper"] = 0
                return command
        
        else :

            if self.state is self.Activity.SEARCHING_WOUNDED and found_wounded:
                self.state = self.Activity.GRASPING_WOUNDED

            elif self.state is self.Activity.GRASPING_WOUNDED and self.base.grasper.grasped_entities:
                self.state = self.Activity.SEARCHING_RESCUE_CENTER

            elif self.state is self.Activity.GRASPING_WOUNDED and not found_wounded:
                self.state = self.Activity.SEARCHING_WOUNDED

            elif self.state is self.Activity.SEARCHING_RESCUE_CENTER and found_rescue_center:
                self.state = self.Activity.DROPPING_AT_RESCUE_CENTER

            elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not self.base.grasper.grasped_entities:
                self.state = self.Activity.SEARCHING_WOUNDED

            elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not found_rescue_center:
                self.state = self.Activity.SEARCHING_RESCUE_CENTER
            """
            print("state: {}, can_grasp: {}, grasped entities: {}".format(self.state.name,
                                                                        self.base.grasper.can_grasp,
                                                                        self.base.grasper.grasped_entities))
            """
            ##########
            # COMMANDS FOR EACH STATE
            # Searching randomly, but when a rescue center or wounded person is detected, we use a special command
            ##########
            if self.state is self.Activity.SEARCHING_WOUNDED:
                command["forward"] = command["forward"] * 0.75 - components[0] / 4
                command["lateral"] = command["lateral"] * 0.75 - components[1] / 4
                command["rotation"] = command["rotation"] * 0.75 + turningComponent * 0.25
                command["grasper"] = 0
                path.append(self.estimated_pose.position)

                #if counter == 1:
                #    self.binaryGrid.grid = self.occupancyGrid.grid.copy()

                self.binaryGrid.grid[self.bonusOccupancyGrid.grid < 0] = -1
                self.binaryGrid.grid[self.bonusOccupancyGrid.grid > 0] = 1

                rows, cols = self.binaryGrid.grid.shape
                for i in range(rows):
                    for j in range(cols):
                        if self.bonusOccupancyGrid.grid[i, j] < 0 and self.binaryGrid.grid[i, j] != 1:
                            self.binaryGrid.grid[i, j] = -1
                        if self.bonusOccupancyGrid.grid[i, j] == 40:
                            self.binaryGrid.grid[i, j] = 1

                self.binaryGrid.display(self.binaryGrid.zoomed_grid, self.estimated_pose, title="zoomed binary grid")
                #print(np.unique(self.binaryGrid.grid))
                self.binaryClone.display(self.binaryClone.zoomed_grid, self.estimated_pose, title="zoomed binary clone")
                
            
                start = self.occupancyGrid._conv_world_to_grid(int(self.estimated_pose.position[0]), int(self.estimated_pose.position[1]))
                
                """
                
                if self.binaryGrid.grid[start[0]][start[1]] == 1 and self.goingBackwards == 0:
                    self.goingBackwards = 10

                if self.goingBackwards > 0:
                    self.goingBackwards -= 1
                    command = {}
                    command["forward"] = -1
                    return(command)
                """

                if self.explorePath == [] or counter % 50 == 0:
                    print('recalculating')
                    start = self.occupancyGrid._conv_world_to_grid(int(self.estimated_pose.position[0]), int(self.estimated_pose.position[1]))
                    self.explorePath = myBrutePather(start, self.binaryGrid.grid)
                    self.binaryClone.grid = self.binaryGrid.grid.copy()
                    for x in self.explorePath:
                        self.binaryClone.grid[x[0]][x[1]] = -40
                    for i in range(len(self.explorePath)):
                        self.explorePath[i] = self.occupancyGrid._conv_grid_to_world(self.explorePath[i][0], self.explorePath[i][1])
                    
                    print(self.estimated_pose.position)
                    print(self.explorePath)
                

                if counter > 40:
                    while self.explorePath != [] and abs(self.explorePath[0][0] - self.estimated_pose.position[0]) < 10 and abs(self.explorePath[0][1] - self.estimated_pose.position[1]) < 10:
                        self.explorePath.pop(0) 
                    if self.explorePath != []:
                        command = self.slideToPlace(self.explorePath[0])
                command["grasper"] = 0
                return command

            elif self.state is self.Activity.GRASPING_WOUNDED:
                command = command_semantic
                command["grasper"] = 1
                path.append(self.estimated_pose.position)

            elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
                goal = self.occupancyGrid._conv_world_to_grid(int(path[0][0]), int(path[0][1]))
                #command = self.control_random()
                #command["forward"] = command["forward"] * 0.75 - components[0] / 4
                #command["lateral"] = command["lateral"] * 0.75 - components[1] / 4
                #command["rotation"] = command["rotation"] * 0.75 + turningComponent * 0.25
                #print(self.astarPath)
                self.binaryGrid.grid[self.bonusOccupancyGrid.grid < 0] = -1
                self.binaryGrid.grid[self.bonusOccupancyGrid.grid > 0] = 1

                rows, cols = self.binaryGrid.grid.shape
                for i in range(rows):
                    for j in range(cols):
                        if self.bonusOccupancyGrid.grid[i, j] <     0 and self.binaryGrid.grid[i, j] != 1:
                            self.binaryGrid.grid[i, j] = -1
                        if self.bonusOccupancyGrid.grid[i, j] == 40:
                            self.binaryGrid.grid[i, j] = 1

                self.binaryGrid.display(self.binaryGrid.zoomed_grid, self.estimated_pose, title="zoomed binary grid")
                #print(np.unique(self.binaryGrid.grid))
                self.binaryClone.display(self.binaryClone.zoomed_grid, self.estimated_pose, title="zoomed binary clone")
                
            
                start = self.occupancyGrid._conv_world_to_grid(int(self.estimated_pose.position[0]), int(self.estimated_pose.position[1]))
                

                """
                if self.binaryGrid.grid[start[0]][start[1]] == 1 and self.goingBackwards == 0:
                    self.goingBackwards = 10

                if self.goingBackwards > 0:
                    self.goingBackwards -= 1
                    command = {}
                    command["forward"] = -1
                    return(command)
                """

                if self.explorePath == [] or counter % 50 == 0:
                    print('recalculating')
                    start = self.occupancyGrid._conv_world_to_grid(int(self.estimated_pose.position[0]), int(self.estimated_pose.position[1]))
                    self.explorePath = self.astarPath
                    self.binaryClone.grid = self.binaryGrid.grid.copy()
                    for x in self.explorePath:
                        self.binaryClone.grid[x[0]][x[1]] = -40
                    for i in range(len(self.explorePath)):
                        self.explorePath[i] = self.occupancyGrid._conv_grid_to_world(self.explorePath[i][0], self.explorePath[i][1])
                    
                    print(self.estimated_pose.position)
                    print(self.explorePath)
                

                if counter > 40:
                    while self.explorePath != [] and abs(self.explorePath[0][0] - self.estimated_pose.position[0]) < 10 and abs(self.explorePath[0][1] - self.estimated_pose.position[1]) < 10:
                        self.explorePath.pop(0) 
                    if self.explorePath != []:
                        command = self.slideToPlace(self.explorePath[0])
                command["grasper"] = 1
                return command

            elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
                command = command_semantic
                command["grasper"] = 1

            return command

    def process_lidar_sensor(self):
        """
        Returns True if the drone collided an obstacle
        """
        if self.lidar_values() is None:
            return False

        collided = False
        dist = min(self.lidar_values())

        if dist < 40:
            collided = True

        return collided

    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move towards a wound person or the rescue center
        """
        command = {"forward": 0.5,
                   "lateral": 0.0,
                   "rotation": 0.0}
        angular_vel_controller_max = 1.0

        detection_semantic = self.semantic_values()
        best_angle = 0

        found_wounded = False
        if (self.state is self.Activity.SEARCHING_WOUNDED
            or self.state is self.Activity.GRASPING_WOUNDED) \
                and detection_semantic is not None:
            scores = []
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    found_wounded = True
                    v = (data.angle * data.angle) + \
                        (data.distance * data.distance / 10 ** 5)
                    scores.append((v, data.angle, data.distance))

            # Select the best one among wounded persons detected
            best_score = 10000
            for score in scores:
                if score[0] < best_score:
                    best_score = score[0]
                    best_angle = score[1]

        found_rescue_center = False
        is_near = False
        angles_list = []
        if (self.state is self.Activity.SEARCHING_RESCUE_CENTER
            or self.state is self.Activity.DROPPING_AT_RESCUE_CENTER) \
                and detection_semantic:
            for data in detection_semantic:
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    found_rescue_center = True
                    angles_list.append(data.angle)
                    is_near = (data.distance < 50)

            if found_rescue_center:
                best_angle = circular_mean(np.array(angles_list))

        if found_rescue_center or found_wounded:
            # simple P controller
            # The robot will turn until best_angle is 0
            kp = 2.0
            a = kp * best_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a * angular_vel_controller_max

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.2

        if found_rescue_center and is_near:
            command["forward"] = 0
            command["rotation"] = random.uniform(0.5, 1)

        return found_wounded, found_rescue_center, command
    
    def process_communication_sensor(self):
        found_drone = False
        command_comm = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0}
        if self.state is self.Activity.SLAVE:
            if self.communicator :
                for data in self.communicator.received_messages:
                    data = data[1]
                    if data[0] == 0:
                        found_drone = True
                        command_comm = self.faceCoordinates(data[1][0])
                        command_comm["grasper"] = 0
                        command_comm["forward"] = 1
                        xd, yd = data[1][0]
                        xs, ys = self.estimated_pose.position
                        distance = math.sqrt((xd - xs)**2 + (yd - ys)**2)
                        if distance < 100:
                            command_comm["forward"] = -1
                            command_comm["rotation"] = 0
                            command_comm["lateral"] = 1
                            command_comm["grasper"] = 0
        return found_drone, command_comm
    


    def getImbalanceList(self):
        #if all three over 150 distanceImbalance is 0
        imbalanceList = [0 for i in range(181)]
        for i in range(len(self.lidar_values())):
            opposing = getOpposingRayIndex(i)
            if min(self.lidar_values()[i],max(self.lidar_values()[opposing[0]],self.lidar_values()[opposing[1]])) > 150:
                imbalanceList[i] = 0
            else : 
                imbalanceList[i] = ((self.lidar_values()[opposing[0]]/2 + self.lidar_values()[opposing[1]]/2) - self.lidar_values()[i])
        self.imbalanceList = imbalanceList

    def faceCoordinates(self, coordinates):
        x1 = self.estimated_pose.position[0]
        y1 = self.estimated_pose.position[1]
        x2 = coordinates[0]
        y2 = coordinates[1]
    # Calculate the angle in radians between two points (x1, y1) and (x2, y2)
        angle = math.atan2(y2 - y1, x2 - x1)
        angle_difference = angle - self.estimated_pose.orientation
        angle_difference = math.atan2(math.sin(angle_difference), math.cos(angle_difference))
        command = {}
        if abs(angle - self.estimated_pose.orientation) > 0.25:
            command["forward"] = 0
        else : 
            command["forward"] = 1
        command["lateral"] = 0
        if angle_difference > 0:
            command["rotation"] = 1
        else : 
            command["rotation"] = -1
        if self.state is self.Activity.SEARCHING_WOUNDED:
            command["grasper"] = 0
        else: 
            command["grasper"] = 1
        return(command)

    def slideToPlace(self, place):
        x1 = self.estimated_pose.position[0]
        y1 = self.estimated_pose.position[1]
        x2 = place[0]
        y2 = place[1]
        #print(x1, ',',y1, " and ",x2, ',',y2)

        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        if distance > 100:
            coeff = 1
        elif distance < 0:
            coeff = 0.2
        else:
            # Linear scaling between 0.2 and 1 for distances between 0 and 1000
            coeff = 0.2 + 0.8 * (distance / 100)

        coeff = 1
        # Calculate angle
        angle = math.atan2(y2 - y1, x2 - x1)

        command = {}
        # Calculate forward and lateral components
        command["forward"] = math.cos(angle - self.estimated_pose.orientation) * coeff
        command["lateral"] = math.sin(angle - self.estimated_pose.orientation) * coeff
        return(command)

    def goToPlace(self, place):
        """
        According to his state in the state machine, the Drone will move towards a wound person or the rescue center
        """
        command = {"forward": 1,
                   "lateral": 0.0,
                   "rotation": 0.0}
        angular_vel_controller_max = 1.0

        detection_semantic = self.semantic_values()
        best_angle = 0

        x1 = self.estimated_pose.position[0]
        y1 = self.estimated_pose.position[1]
        x2 = place[0]
        y2 = place[1]
    # Calculate the angle in radians between two points (x1, y1) and (x2, y2)
        angle = math.atan2(y2 - y1, x2 - x1)

        relativeAngle = angle - self.estimated_pose.orientation

        found_rescue_center = False
        is_near = False
        angles_list = []
        # simple P controller
        # The robot will turn until best_angle is 0
        kp = 2.0
        a = kp * relativeAngle
        a = min(a, 1.0)
        a = max(a, -1.0)
        command["rotation"] = a * angular_vel_controller_max

        # reduce speed if we need to turn a lot
        if abs(a) == 1:
            command["forward"] = 0.2

        return command


def getOpposingRayIndex(index):
    opposing_index1 = (index + 181 // 2) % 181
    return (opposing_index1%181, (opposing_index1 + 1)%181)

def calculate_movement_components(rayNumber):
    angle_radians = math.radians(rayNumber * 2)

    # Calculate forward and lateral components
    forward = math.cos(angle_radians)
    lateral = math.sin(angle_radians)

    return forward, lateral

def astar(graph, start, goal):
    def heuristic(node, goal):
        return euclidean(node, goal)

    open_set = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        current_cost, current_node = heappop(open_set)

        if current_node == goal:
            path = reconstruct_path(came_from, start, goal)
            return path

        for neighbor in graph.neighbors(current_node):
            new_cost = cost_so_far[current_node] + 1  # Assuming each edge has a cost of 1

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current_node

    return None

def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]

    while current != start:
        current = came_from[current]
        path.append(current)

    return path[::-1]

def myBrutePather(start, grid):
    graph = {}
    m = len(grid)
    n = len(grid[0])
# Iterate through the grid
    for i in range(m):
        for j in range(n):
            value = grid[i][j]
            # If the value is -1, create edges to adjacent -1s and 0s
            if value == -1 or value == 1:
                neighbors = []

                # Check left neighbor
                if j > 0 and grid[i][j - 1] in (-1, 0):
                    neighbors.append((i, j - 1))

                # Check right neighbor
                if j < n - 1 and grid[i][j + 1] in (-1, 0):
                    neighbors.append((i, j + 1))

                # Check upper neighbor
                if i > 0 and grid[i - 1][j] in (-1, 0):
                    neighbors.append((i - 1, j))

                # Check lower neighbor
                if i < m - 1 and grid[i + 1][j] in (-1, 0):
                    neighbors.append((i + 1, j))

                # Add edges to the graph
                graph[(i, j)] = neighbors

            # If the value is 0, mark it as a goal node
            elif value == 0:
                # You can create a special node for goal nodes or mark it as needed
                graph[(i, j)] = "Goal Node"
    

    theSet = {start: start}
    notFound = True
#print(graph)
    counter = n*m
    while notFound and counter > 0:
        #print(counter)
        counter -= 1
        newSet = {}
        for coords, predecessor in theSet.items():
            #print(coords)
            #print(grid[coords[0]][coords[1]])
            if graph[coords] == "Goal Node":
                goal = coords
                notFound = False
                break
            else : 
                for x in graph[coords]:
                    if not x in theSet:
                        newSet[x] = coords
        theSet.update(newSet)
    
    if notFound:
        return []
    else:
        path = [goal]
        #print(start)
        while path[-1] != start:
            #print(path[-1])
            path.append(theSet[path[-1]])
        path.reverse()
    return path

def myBruteGoTo(start, grid, end):
    graph = {}
    m = len(grid)
    n = len(grid[0])
    #print("n et m : ", n, m)
    #print(grid)
# Iterate through the grid
    for i in range(m):
        for j in range(n):
            value = grid[i][j]
            # If the value is -1, create edges to adjacent -1s and 0s
            if abs(value) == 1 :
                neighbors = []

                # Check left neighbor
                if j > 0 and grid[i][j - 1] in (-1, 0):
                    neighbors.append((i, j - 1))

                # Check right neighbor
                if j < n - 1 and grid[i][j + 1] in (-1, 0):
                    neighbors.append((i, j + 1))

                # Check upper neighbor
                if i > 0 and grid[i - 1][j] in (-1, 0):
                    neighbors.append((i - 1, j))

                # Check lower neighbor
                if i < m - 1 and grid[i + 1][j] in (-1, 0):
                    neighbors.append((i + 1, j))

                # Add edges to the graph
                graph[(i, j)] = neighbors

            # If the value is 0, mark it as a goal node
            #elif value == 0:
                # You can create a special node for goal nodes or mark it as needed
                
    graph[(end[0], end[1])] = "Goal Node"
    #print(end[0],end[1])
    
    theSet = {start: start}
    notFound = True
#print(graph)
    counter = n*m
    while notFound and counter > 0:
        #print(counter)
        counter -= 1
        newSet = {}
        for coords, predecessor in theSet.items():
            #print(coords)
            #print(grid[coords[0]][coords[1]])
            if coords in graph.keys() :
                if graph[coords] == "Goal Node":
                    goal = coords
                    notFound = False
                    break
                else : 
                    for x in graph[coords]:
                        if not x in theSet:
                            newSet[x] = coords
        theSet.update(newSet)
    
    if notFound:
        return []
    else:
        path = [goal]
        #print(start)
        while path[-1] != start:
            #print(path[-1])
            path.append(theSet[path[-1]])
        path.reverse()
    return path



class MyMapSemantic(MapAbstract):
    def __init__(self):
        super().__init__()

        # PARAMETERS MAP
        self._size_area = (400, 400)

        self._rescue_center = RescueCenter(size=(100, 100))
        self._rescue_center_pos = ((0, 150), 0)

        # WOUNDED PERSONS
        self._number_wounded_persons = 20
        self._wounded_persons_pos = []
        self._wounded_persons: List[WoundedPerson] = []

        start_area = (0.0, -30.0)
        nb_per_side = math.ceil(math.sqrt(float(self._number_wounded_persons)))
        dist_inter_wounded = 60.0
        sx = start_area[0] - (nb_per_side - 1) * 0.5 * dist_inter_wounded
        sy = start_area[1] - (nb_per_side - 1) * 0.5 * dist_inter_wounded

        for i in range(self._number_wounded_persons):
            x = sx + (float(i) % nb_per_side) * dist_inter_wounded
            y = sy + math.floor(float(i) / nb_per_side) * dist_inter_wounded
            pos = ((x, y), random.uniform(-math.pi, math.pi))
            self._wounded_persons_pos.append(pos)

        # POSITIONS OF THE DRONES
        self._number_drones = 1
        self._drones_pos = [((40, 40), random.uniform(-math.pi, math.pi))]
        self._drones = []

    def construct_playground(self, drone_type: Type[DroneAbstract]):
        playground = ClosedPlayground(size=self._size_area)

        # RESCUE CENTER
        playground.add_interaction(CollisionTypes.GEM,
                                   CollisionTypes.ACTIVABLE_BY_GEM,
                                   wounded_rescue_center_collision)

        playground.add(self._rescue_center, self._rescue_center_pos)

        # POSITIONS OF THE WOUNDED PERSONS
        for i in range(self._number_wounded_persons):
            wounded_person = WoundedPerson(rescue_center=self._rescue_center)
            self._wounded_persons.append(wounded_person)
            pos = self._wounded_persons_pos[i]
            playground.add(wounded_person, pos)

        # POSITIONS OF THE DRONES
        misc_data = MiscData(size_area=self._size_area,
                             number_drones=self._number_drones)
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            playground.add(drone, self._drones_pos[i])

        return playground


def main():
    my_map = MyMapSemantic()
    playground = my_map.construct_playground(drone_type=MyDroneSemantic)

    # draw_semantic_rays : enable the visualization of the semantic rays
    gui = GuiSR(playground=playground,
                the_map=my_map,
                draw_semantic_rays=True,
                use_keyboard=False,
                )
    gui.run()


if __name__ == '__main__':
    main()
