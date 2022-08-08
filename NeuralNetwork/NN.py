import torch
import numpy as np
from torchvision.transforms import ToTensor, Lambda
from torch import nn


class ValueNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ValueNN, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, output_size),

        )

    def forward(self, input_element):
        output = self.linear_relu_stack(input_element)
        return output


class ProbNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ProbNN, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, input_element):
        output = self.linear_relu_stack(input_element)
        return output


class HopeNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(HopeNN, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, input_element):
        output = self.linear_relu_stack(input_element)
        return output

