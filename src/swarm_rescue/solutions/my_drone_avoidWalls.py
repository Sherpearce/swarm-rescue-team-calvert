"""
Simple random controller
The Drone will move forward and turn for a random angle when an obstacle is hit
"""
import math
import random
from typing import Optional

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle


class MyDroneAvoidWalls(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)
        self.counterStraight = 0
        self.angleStopTurning = random.uniform(-math.pi, math.pi)
        self.distStopStraight = random.uniform(10, 50)
        self.isTurning = False




        self.goingForPerson = False
        self.randomMovement = False

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def process_lidar_sensor(self):
        """
        Returns True if the drone collided an obstacle
        """
        if self.lidar_values() is None:
            return False

        collided = False
        dist = min(self.lidar_values())
        print(self.lidar_values())
        print(len(self.lidar_values()))

        if dist < 40:
            collided = True

        return collided

    def control(self):
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit
        """
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

        if collided and not self.isTurning and self.counterStraight > self.distStopStraight:
            self.isTurning = True
            self.angleStopTurning = random.uniform(-math.pi, math.pi)

        measured_angle = 0
        if self.measured_compass_angle() is not None:
            measured_angle = self.measured_compass_angle()

        diff_angle = normalize_angle(self.angleStopTurning - measured_angle)
        if self.isTurning and abs(diff_angle) < 0.2:
            self.isTurning = False
            self.counterStraight = 0
            self.distStopStraight = random.uniform(10, 50)

        if self.isTurning:
            turning = 1
            forwarding = 0
        else:
            turning = 0
            forwarding = 1

            #we want to be 150 away from walls to be far away but to see the bodies


        custom_command = {
            "forward": forwarding*0.75 - components[0] / 4,
            "lateral": - components[1] / 4,
            "rotation": turning * 0.75 + turningComponent * 0.25,
            "grasper": 0
        }

        return custom_command
            #we will make a liste of difference of distance in between ray and 2 opposing rays
            # (if one of the rays is under 150)
          
            
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

def getOpposingRayIndex(index):
    opposing_index1 = (index + 181 // 2) % 181
    return (opposing_index1%181, (opposing_index1 + 1)%181)

def calculate_movement_components(rayNumber):
    angle_radians = math.radians(rayNumber * 2)

    # Calculate forward and lateral components
    forward = math.cos(angle_radians)
    lateral = math.sin(angle_radians)

    return forward, lateral