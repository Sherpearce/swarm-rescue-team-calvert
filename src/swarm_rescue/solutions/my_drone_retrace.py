"""
This program can be launched directly.
Example of how to use semantic sensor, grasping and dropping
"""

import os
import sys
import random
import math
from typing import Optional, List, Type
from enum import Enum

import numpy as np
from spg.utils.definitions import CollisionTypes

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

counter = 0
path = []

class MyDroneRetrace(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4

    def __init__(self,
                 identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         display_lidar_graph=False,
                         **kwargs)
        # The state is initialized to searching wounded person
        self.state = self.Activity.SEARCHING_WOUNDED

        # Those values are used by the random control function
        self.counterStraight = 0
        self.angleStopTurning = 0
        self.isTurning = False
        self.tryingToFace = False

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def control(self):
        global counter, path
        counter += 1
        print(counter)
        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()), self.measured_compass_angle())
        print(self.estimated_pose.position)
        print(self.estimated_pose.orientation)


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
            command = self.control_random()
            command["forward"] = command["forward"] * 0.75 - components[0] / 4
            command["lateral"] = command["lateral"] * 0.75 - components[1] / 4
            command["rotation"] = command["rotation"] * 0.75 + turningComponent * 0.25
            command["grasper"] = 0
            path.append(self.estimated_pose.position)

        elif self.state is self.Activity.GRASPING_WOUNDED:
            command = command_semantic
            command["grasper"] = 1
            path.append(self.estimated_pose.position)

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            #command = self.control_random()
            #command["forward"] = command["forward"] * 0.75 - components[0] / 4
            #command["lateral"] = command["lateral"] * 0.75 - components[1] / 4
            #command["rotation"] = command["rotation"] * 0.75 + turningComponent * 0.25
            while abs(path[-1][0] - self.estimated_pose.position[0]) < 40 and abs(path[-1][1] - self.estimated_pose.position[1]) < 40:
                path.pop(-1) 
            command = self.goToPlace(path[-1])
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

    def control_random(self):
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit
        """
        command_straight = {"forward": 1.0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 0}


        command_turn = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 1.0,
                        "grasper": 0}

        collided = self.process_lidar_sensor()

        self.counterStraight += 1

        if collided and not self.isTurning and self.counterStraight > 20:
            self.isTurning = True
            self.angleStopTurning = random.uniform(-math.pi, math.pi)

        diff_angle = normalize_angle(
            self.angleStopTurning - self.measured_compass_angle())
        if self.isTurning and abs(diff_angle) < 0.2:
            self.isTurning = False
            self.counterStraight = 0

        if self.isTurning:
            return command_turn
        else:
            return command_straight

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
