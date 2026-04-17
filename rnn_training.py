# contain main model training function
import torch
import torch.nn as nn

import time
from torch.utils.data import DataLoader
from load_data import *

def train_model(model,
                dataloader, 
                device,
                report_every=5,
                num_epochs=10, 
                batch_size=32, 
                learning_rate=0.01):

    # loss book keeping
    current_loss = 0
    all_losses = []
    total_samples = 0

    model.to(device)

    # define optimizer and criterion (loss fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    start = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        model.train()

        # for each batch in the dataloader
        for y_batch, X_batch in dataloader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # process da batch
            output = model(X_batch)
            loss = criterion(output, y_batch)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

            current_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

        # loss book keeping calculation
        all_losses.append(current_loss / total_samples)

        if epoch % report_every == 0:
            print(f"{epoch} ({epoch / num_epochs:.0%}): \t average batch loss = {all_losses[-1]}")

        current_loss = 0

    end = time.perf_counter()
    total_time = end - start

    results = {
        "learning_rate" : learning_rate,
        "batch_size"    : batch_size,
        "num_epochs"    : num_epochs,
        "Training Time" : total_time,
        "All Losses"    : all_losses
    }

    return results



# def train(num_epochs, input_size, hidden_size, output_size):
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


# def train(rnn, training_data, n_epoch = 10, n_batch_size = 64, report_every = 50, learning_rate = 0.2, criterion = nn.NLLLoss()):
#     """
#     Learn on a batch of training_data for a specified number of iterations and reporting thresholds
#     """
#     # Keep track of losses for plotting
#     current_loss = 0
#     all_losses = []
#     rnn.train()
#     optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

#     start = time.perf_counter()
#     print(f"training on data set with n = {len(training_data)}")

#     # create some minibatches
#     loader = DataLoader(training_data, batch_size=n_batch_size, shuffle=True)

#     for iter in range(1, n_epoch + 1):
#         rnn.zero_grad() # clear the gradients


#         for y_batch, X_batch in loader:
#             optimizer.zero_grad()

#             batch_loss = 0

#             # If your RNN expects one sequence at a time:
#             for i in range(len(X_batch)):
#                 output = rnn(X_batch[i])
#                 loss = criterion(output, y_batch[i])
#                 batch_loss += loss

#             # optimize parameters
#             batch_loss.backward()
#             nn.utils.clip_grad_norm_(rnn.parameters(), 3)
#             optimizer.step()
#             optimizer.zero_grad()

#             current_loss += batch_loss.item() / len(X_batch)

#         all_losses.append(current_loss / len(loader) )
#         if iter % report_every == 0:
#             print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
#         current_loss = 0

#     return all_losses

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