import torch

class RNNGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNGRU, self).__init__()

        self.rnn_process = nn.Sequential(

            nn.GRU(input_size, hidden_size),

            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, line_tensor):
        output = self.rnn_process(line_tensor)
        return output

