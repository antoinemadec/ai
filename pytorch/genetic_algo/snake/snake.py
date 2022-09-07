#!/usr/bin/env python3

import random
import numpy as np
from numpy.typing import NDArray


class Snake:
    """Snake game class"""

    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        x = self.grid_size//2
        self.snake_coords = [(x, x-1), (x, x), (x, x+1)]
        self.set_new_fruit()
        self.step_nb = 0

    def coord_is_wall(self, coord):
        return coord[0] == 0 or coord[1] == 0 or coord[0] == self.grid_size-1 or coord[1] == self.grid_size-1

    def set_new_fruit(self):
        self.fruit_coords = self.snake_coords[-1]
        while self.fruit_coords in self.snake_coords:
            x = random.randrange(1, self.grid_size-1)
            y = random.randrange(1, self.grid_size-1)
            self.fruit_coords = (x, y)

    def process_next_state(self, direction: str) -> bool:
        self.step_nb += 1

        if direction == "right":
            d = (0, 1)
        elif direction == "left":
            d = (0, -1)
        elif direction == "up":
            d = (-1, 0)
        elif direction == "down":
            d = (1, 0)
        else:
            raise ValueError("direction must be in right/left/up/down")

        # compute new head
        head = self.snake_coords[-1]
        new_head = (head[0] + d[0], head[1] + d[1])
        if new_head in self.snake_coords or self.coord_is_wall(new_head):
            return False

        # update snake
        self.snake_coords += [new_head]
        if new_head != self.fruit_coords:
            self.snake_coords = self.snake_coords[1:-1] + [new_head]
        else:
            self.set_new_fruit()

        return True

    def get_grid(self) -> NDArray[np.float32]:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        # walls
        grid[0] = 1
        grid[self.grid_size-1] = 1
        grid[:, 0] = 1
        grid[:, self.grid_size-1] = 1

        snake_coords = np.array(self.snake_coords)
        grid[tuple(snake_coords.T)] = 2
        grid[self.snake_coords[-1]] = 3
        grid[self.fruit_coords] = 4
        return grid

    def print_grid(self) -> str:
        string = ""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if x == 0 or x == self.grid_size-1:
                    string += "_"
                elif y == 0 or y == self.grid_size-1:
                    string += "|"
                elif (x, y) in self.snake_coords:
                    if (x, y) == self.snake_coords[-1]:
                        string += "@"
                    else:
                        string += "*"
                elif (x, y) == self.fruit_coords:
                    string += "O"
                else:
                    string += " "
            string += "\n"
        return string

#     s = Snake()
#     while True:
#         s.print_grid()
#         time.sleep(0.2)
#         direction = keys_q.get()
#         if keys_q.empty():
#             keys_q.put(direction)
#         s.process_next_state(direction)
