import torch
import torch.nn as nn

class RNNGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNGRU, self).__init__()

        self.rnn = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        self.h2o = nn.Linear(hidden_size, output_size)


    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        return output

