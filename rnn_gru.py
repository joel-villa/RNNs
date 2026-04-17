import torch
import torch.nn as nn

class RNNGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNGRU, self).__init__()

        self.rnn = nn.GRU(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        output = self.softmax(output)
        return output

