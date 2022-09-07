#!/usr/bin/env python3

import torch
from torch import nn

from snake.snake import Snake  # type: ignore

# -------------------------------------------------------------
# model
# -------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


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


def infer_direction(grid, model: NeuralNetwork):
    with torch.no_grad():
        pred = model(torch.from_numpy(grid).to(device))
        index = pred.argmax(0)
        direction = ("right", "left", "up", "down")[index]
        print(direction)
        return direction


with torch.no_grad():
    model = NeuralNetwork().to(device)
    print(model)
    model.eval()

    s = Snake()

    while s.process_next_state(infer_direction(s.get_grid(), model)):
        # print(s.print_grid())
        print(s.get_score())

    # TODO: cross over
    # for p in model.parameters():
    #     p_flat = p.flatten()
        # for i, v in enumerate(p_flat):
        #     p_flat[i] = 0
