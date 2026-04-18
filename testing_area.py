# do model training and testing here
import torch
from torch.utils.data import DataLoader

from rnn_training import train_model
from rnn_baseline import RNNBaseline
from rnn_gru import RNNGRU
from rnn_lstm import RNNLSTM
from load_data import *
from rnn_testing import test_model

BATCH_SIZE = 32
REPORT_EVERY = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"We are using: {device}")

def parameter_search(
        train_dataloader,
        test_set,
        num_hidden_layers,
        num_hidden_neurons,
        learning_rates,
        model_type="rnn_baseline", # could either be rnn_normal, rnn_gru, rnn_lstm
        num_epochs=50,
        ticker="AAPL"):

    best_results = {
        "MAE"           : 100.0,
        "MAPE"          : 100.0,
        "Details"       : None,
        "Ticker"        : ticker,
        "model_type"    : model_type
    }

    match model_type:
        case "rnn_baseline":
            model_thing = RNNBaseline
        case "rnn_gru":
            model_thing = RNNGRU
        case "rnn_lstm":
            model_thing = RNNLSTM
        case _:
            print(f"Yo make sure this model exists: {model_type}")
            return None

    for layers in num_hidden_layers:
        for neurons in num_hidden_neurons:
            for learning_rate in learning_rates:
                # create a model
                model = model_thing(
                                input_size=1,
                                output_size=1,
                                num_layers=layers,
                                hidden_size=neurons)

                details = train_model(
                    model=model, 
                    dataloader=train_dataloader, 
                    device=device,
                    report_every=REPORT_EVERY,
                    num_epochs=num_epochs, 
                    learning_rate=learning_rate
                )

                mae, mape = test_model(
                    model=model,
                    test_set=test_set, 
                    device=device,
                    ticker=ticker
                )

                if best_results["MAPE"] > mape:
                    best_results = {
                        "MAE"           : mae,
                        "MAPE"          : mape,
                        "Details"       : details,
                        "Ticker"        : ticker,
                        "model_type"    : model_type
                    }

    return best_results

def old_main():
    print("Starting!")

    # RNN INIT
    rnn_baseline = RNNBaseline(input_size=1, hidden_size=16, output_size=1, num_layers=1)

    train_set, test_set = load_AAPL()

    label, x = train_set[0]
    print(x.shape)

    train_dataloader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    results = train_model(
        model=rnn_baseline, 
        dataloader=train_dataloader, 
        device=device,
        report_every=1,
        num_epochs=10, 
        learning_rate=0.1e-3
    )

    print(results)

    test_model(model=rnn_baseline,
            test_set=test_set, 
            device=device,
            ticker="AAPL")

if __name__ == "__main__":
   
    train_set, test_set = load_AAPL()

    train_dataloader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    ticker = "AAPL"
    model_type = "rnn_baseline"

    num_hidden_layers = [1]
    num_hidden_neurons = [1]
    learning_rates = [1e-1]

    best_result = parameter_search(
        train_dataloader=train_dataloader,
        test_set=test_set,
        num_hidden_layers=num_hidden_layers,
        num_hidden_neurons=num_hidden_neurons,
        learning_rates=learning_rates,
        model_type=model_type,
        num_epochs=20,
        ticker=ticker)

    print(best_result)







