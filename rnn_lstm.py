import torch
import torch.nn as nn

class RNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLSTM, self).__init__()

        self.rnn_process = nn.Sequential(

            nn.LSTM(input_size, hidden_size),

            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, line_tensor):
        output = self.rnn_process(line_tensor)
        return output

