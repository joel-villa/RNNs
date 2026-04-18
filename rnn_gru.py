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
        # take last val, same as hidden[0] 
        last_val = rnn_out[:, -1, :]
        output = self.h2o(last_val)
        return output
    
    def predict(self, X):
        self.eval()  # set to evaluation mode

        with torch.no_grad():  # disable gradients
            output = self.forward(X)

        return output

