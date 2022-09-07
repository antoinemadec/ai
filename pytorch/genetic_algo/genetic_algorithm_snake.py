#!/usr/bin/env python3

import torch
from torch import nn
import numpy as np

from snake.snake import Snake  # type: ignore


class NeuralNetwork(nn.Module):
    def __init__(self, grid_size=20):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(grid_size*grid_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = x.flatten()
        logits = self.linear_relu_stack(x)
        return logits


class Individual:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, grid_size=20):
        with torch.no_grad():
            self.model = NeuralNetwork().to(self.device)
            self.model.eval()
        self.score = 0

    def infer_direction(self, grid):
        with torch.no_grad():
            pred = self.model(torch.from_numpy(grid).to(self.device))
            index = pred.argmax(0)
            direction = ("right", "left", "up", "down")[index]
            return direction

    def play(self) -> None:
        s = Snake()
        while s.process_next_state(self.infer_direction(s.get_grid())):
            if (s.step_nb > 150000):
                break
        self.score = len(s.snake_coords)


population: list[Individual] = []

# 1- initial population
for i in range(1000):
    population.append(Individual())

for gen_idx in range(100):
    print(f"gen_{gen_idx}")
    # 2- evaluation
    for individual in population:
        individual.play()

    # 3- selection
    population.sort(key=lambda i: i.score, reverse=True)
    parent = population[0]
    print(f"best={parent.score}")

    # 4- crossover
    with torch.no_grad():
        for individual in population[1:]:
            for parent_p, p in zip(parent.model.parameters(), individual.model.parameters()):
                parent_p_flat = torch.clone(parent_p.flatten())
                p_flat = p.flatten()
                mask = np.random.randint(2, size=p_flat.shape[0])
                p_flat[mask == 1] = parent_p_flat[mask == 1]

    # 5- mutation
