import numpy as np
import pandas as pd

class DiscreteVaccumRobotV0:
    
    def __init__(self, room_file):
        self.room_data = pd.read_csv(room_file, header=None).astype(float)
        self.room_data = self.room_data.values
        not_visited_slots = np.where(self.room_data == 1)
        initial_pos_index = np.random.randint(0, len(not_visited_slots[0]))
        self.position = [not_visited_slots[0][initial_pos_index],
                         not_visited_slots[1][initial_pos_index]]
        self.rotation = np.random.choice([0, 90, 180, 270])
        self.room_data[self.position[0]][self.position[1]] = -0.1
        
    def step(self, action):
        # Move forward
        if action == 0:
            if self.rotation == 0:
                next_position = [self.position[0] - 1, self.position[1]]
            elif self.rotation == 90:
                 next_position = [self.position[0], self.position[1] + 1]
            elif self.rotation == 180:
                 next_position = [self.position[0] + 1, self.position[1]]
            elif self.rotation == 270:
                 next_position = [self.position[0], self.position[1] - 1]
            if self.room_data[next_position[0]][next_position[1]] != 2:
                reward = self.room_data[next_position[0]][next_position[1]]
                self.position = next_position
                self.room_data[next_position[0]][next_position[1]] = -0.1
                found_obstacle = False
            else:
                reward = -0.1
                found_obstacle = True
        else:
            # Rotate
            reward = -0.1
            self.rotation += 90 if action == 1 else -90
            if self.rotation == -90:
                self.rotation = 270
            elif self.rotation == 360:
                self.rotation = 0
            found_obstacle = False
        return [[self.position[0], self.position[1], self.rotation,
                 found_obstacle], reward, (self.room_data == 1).any() == False,
                {}]
    
        
robot_env = DiscreteVaccumRobotV0('room1.csv')
print(robot_env.room_data)