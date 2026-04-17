import torch
import torch.nn as nn

class RNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNLSTM, self).__init__()

        self.rnn = self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.h2o = nn.Linear(hidden_size, output_size)


    def forward(self, line_tensor):
        rnn_out, (hidden, cell) = self.rnn(line_tensor)
        # take last val, same as hidden[0] 
        last_out = rnn_out[:, -1, :]
        output = self.h2o(last_out)
        return output
