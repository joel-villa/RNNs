import torch
import torch.nn as nn

from torch.utils.data import DataLoader

# from rnn_training import train_model
from rnn_baseline import RNNBaseline
from rnn_gru import RNNGRU
from rnn_lstm import RNNLSTM
from load_data import *

import matplotlib.pyplot as plt

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model,
                dataloader, 
                device,
                report_every=5,
                num_epochs=10, 
                batch_size=32, 
                learning_rate=0.01): #TODO: stop using this, bad coding practices

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


def plot_training_losses(loss_dict, type):
    plt.figure(figsize=(16, 8))

    for ticker, losses in loss_dict.items():
        plt.plot(range(1, len(losses) + 1), losses, label=ticker)

    plt.title(f"Training Loss Over Epochs ({type})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.savefig(f"plots/{type.replace(" ", "_")}.jpg")
    plt.show()

if __name__ == "__main__":
    stocks = ["AAPL", "MSFT", "TSLA", "NDAQ"]

    best_params = {
        "rnn_baseline" : (3, 512, 0.0001),
        "rnn_gru"      : (3, 256, 0.001),
        "rnn_lstm"     : (2, 256, 0.001)
    }

    BATCH_SIZE = 32

    for model_type in ["rnn_baseline", "rnn_gru","rnn_lstm"]:
        match model_type:
            case "rnn_baseline":
                model_thing = RNNBaseline
            case "rnn_gru":
                model_thing = RNNGRU
            case "rnn_lstm":
                model_thing = RNNLSTM
            case _:
                print(f"Yo make sure this model exists: {model_type}")

        n_layers, h_size, lr = best_params[model_type]

        losses = {}

        for ticker in stocks:
            train_set, test_set = npz_load(ticker=ticker)

            train_dataloader = DataLoader(
                train_set,
                batch_size=BATCH_SIZE,
                shuffle=True
            )
        
            

            model = model_thing(input_size=1,
                                output_size=1, 
                                num_layers=n_layers, 
                                hidden_size=h_size)
            
            results = train_model(model,
                                  train_dataloader, 
                                  device, 
                                  report_every=1, 
                                  num_epochs=25, #TODO large test with best hyperparams
                                  learning_rate=0.001)
            
            losses[ticker] = results["All Losses"]
        
        match model_type:
            case "rnn_baseline":
                name = "RNN Baseline"
            case "rnn_gru":
                name = "RNN GRU"
            case "rnn_lstm":
                name = "RNN LSTM"
            case _:
                print(f"Yo make sure this model exists: {model_type}") 

        plot_training_losses(losses, name)
            
            

    # best_result = parameter_search(
    #     train_dataloader=train_dataloader,
    #     test_set=test_set,
    #     num_hidden_layers=num_hidden_layers,
    #     num_hidden_neurons=num_hidden_neurons,
    #     learning_rates=learning_rates,
    #     model_type=model_type,
    #     num_epochs=100,
    #     ticker=ticker)

    # print(best_result)