#!/usr/bin/env python3

import argparse

import numpy as np
import torch
from torch import nn

from snake.snake import Snake  # type: ignore


GRID_SIZE = 20
POPULATION_SIZE = 1000
PARENT_NB = 10

class NeuralNetwork(nn.Module):
    def __init__(self, grid_size=GRID_SIZE):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(grid_size*grid_size, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 4)
        )

    def forward(self, x):
        x = x.flatten()
        logits = self.linear_relu_stack(x)
        return logits


class Individual:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, grid_size=GRID_SIZE):
        with torch.no_grad():
            self.model = NeuralNetwork(grid_size).to(self.device)
            self.model.eval()
        self.score = 0

    def infer_direction(self, grid):
        with torch.no_grad():
            pred = self.model(torch.from_numpy(grid).to(self.device))
            index = pred.argmax(0)
            direction = ("right", "left", "up", "down")[index]
            return direction

    def play(self, display=False) -> None:
        s = Snake()
        self.score = 0
        while s.process_next_state(self.infer_direction(s.get_grid())):
            if display:
                print(s.print_grid())
            if (s.step_nb > 150000):
                return
        snake_len = len(s.snake_coords)
        self.score = snake_len**snake_len + s.step_nb


parser = argparse.ArgumentParser(
    description='Genetic algorithm training NN to play snake')
parser.add_argument(
    '--play', help='play snake with best pre-trained network', action='store_true')
args = parser.parse_args()

if args.play:
    i = Individual()
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))
    i.model = model
    i.play(True)
else:
    population: list[Individual] = []

    # 1- initial population
    for i in range(POPULATION_SIZE):
        population.append(Individual())

    gen_idx = 0
    max_score = 0
    while True:
        # 2- evaluation
        for individual in population:
            individual.play()

        # 3- selection
        population.sort(key=lambda i: i.score, reverse=True)
        parent = population[0]
        print(f"{gen_idx}, {parent.score}")
        if parent.score > max_score:
            max_score = parent.score
            torch.save(parent.model.state_dict(), "model.pth")
        gen_idx += 1

        with torch.no_grad():
            param_equal_cnt = 0
            for individual in population[PARENT_NB:]:
                for parent_p, p in zip(parent.model.parameters(), individual.model.parameters()):
                    # 4- crossover
                    parent_p_flat = torch.clone(parent_p.flatten())
                    p_flat = p.flatten()
                    if p_flat.equal(parent_p_flat):
                        param_equal_cnt += 1
                    mask = np.random.randint(2, size=p_flat.shape[0])
                    p_flat[mask == 1] = parent_p_flat[mask == 1]
                    # 5- mutation
                    mask = np.random.randint(1000, size=p_flat.shape[0])
                    mutated_values = np.random.random(
                        size=p_flat.shape[0]).astype(np.float32)
                    p_flat[mask < 5] = torch.from_numpy(
                        mutated_values)[mask < 5]
            print(f"param_equal_cnt {param_equal_cnt}")
