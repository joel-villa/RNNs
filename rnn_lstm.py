import torch
import torch.nn as nn

class RNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)


    def forward(self, line_tensor):
        rnn_out, (hidden, cell) = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        return output

