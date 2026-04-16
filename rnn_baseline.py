import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from load_data import *

class RNNBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        #TODO: Idk if this is correct, but idk anything at all
        super(RNNBaseline, self).__init__()

        self.rnn_process = nn.Sequential(
            
            nn.RNN(input_size, hidden_size),

            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, line_tensor):
        output = self.rnn_process(line_tensor)
        return output


def train(num_epochs, input_size, hidden_size, output_size):
    for idx, batch in enumerate(batches):
        batch_loss = 0
        for i in batch:
            (label_tensor, text_tensor, label, text) = training_data[i]
            output = rnn.forward(text_tensor) # compute the prediction
            loss = L(output, label_tensor) # compute the loss for a single sample
            batch_loss += loss # sum of the loss
        batch_loss.backward() # compute the gradient
        nn.utils.clip_grad_norm_(rnn.parameters(), 3) # clip the gradient
        optimizer.step()
        optimizer.zero_grad()


def train(rnn, training_data, n_epoch = 10, n_batch_size = 64, report_every = 50, learning_rate = 0.2, criterion = nn.NLLLoss()):
    """
    Learn on a batch of training_data for a specified number of iterations and reporting thresholds
    """
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    rnn.train()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    start = time.time()
    print(f"training on data set with n = {len(training_data)}")

    # create some minibatches
    loader = DataLoader(training_data, batch_size=n_batch_size, shuffle=True)

    for iter in range(1, n_epoch + 1):
        rnn.zero_grad() # clear the gradients


        for y_batch, X_batch in loader:
            optimizer.zero_grad()

            batch_loss = 0

            # If your RNN expects one sequence at a time:
            for i in range(len(X_batch)):
                output = rnn(X_batch[i])
                loss = criterion(output, y_batch[i])
                batch_loss += loss

            # optimize parameters
            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(X_batch)

        all_losses.append(current_loss / len(loader) )
        if iter % report_every == 0:
            print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
        current_loss = 0

    return all_losses

# def sample_code():
#     for idx, batch in enumerate(batches):
#         batch_loss = 0
#         for i in batch:
#             (label_tensor, text_tensor, label, text) = training_data[i]
#             output = rnn.forward(text_tensor) # compute the prediction
#             loss = L(output, label_tensor) # compute the loss for a single sample
#             batch_loss += loss # sum of the loss
#         batch_loss.backward() # compute the gradient
#         nn.utils.clip_grad_norm_(rnn.parameters(), 3) # clip the gradient
#         optimizer.step()
#         optimizer.zero_grad()

# Input dimensions:
rnn = RNNBaseline(1, 16, 1)

train_set, test_set = load_AAPL()

start = time.time()
all_losses = train(rnn, train_set, n_epoch=30, learning_rate=0.15, report_every=5)
end = time.time()
print(f"training took {end-start}s")