import torch
import torch.nn as nn

class RNNBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNBaseline, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)


    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        return output