import numpy as np
import pandas as pd

class DiscreteVaccumRobotV0:
    
    def __init__(self, room_file, max_moves):
        self.room_file = room_file
        self.max_moves = max_moves

    def reset(self):
        self.room_data = pd.read_csv(self.room_file, header=None).astype(float)
        self.room_data = self.room_data.values
        self.room_data_shape = self.room_data.shape
        not_visited_slots = np.where(self.room_data == 1)
        initial_pos_index = np.random.randint(0, len(not_visited_slots[0]))
        self.position = [not_visited_slots[0][initial_pos_index],
                         not_visited_slots[1][initial_pos_index]]
        self.rotation = np.random.choice([0, 90, 180, 270])
        self.room_data[self.position[0]][self.position[1]] = -0.1
        self.current_move = 0
        return [self.position[0], self.position[1], self.rotation, False]
        
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
        self.current_move += 1
        return [[self.position[0], self.position[1], self.rotation,
                 found_obstacle], reward, (self.room_data == 1).any() == False or self.current_move == self.max_moves,
                {}]
    

if __name__ == '__main__':
    robot_env = DiscreteVaccumRobotV0('room1.csv', 100)
    print(robot_env.room_data)