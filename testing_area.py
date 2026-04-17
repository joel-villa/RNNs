# do model training and testing here
import torch
from torch.utils.data import DataLoader

from rnn_training import train_model
from rnn_baseline import RNNBaseline
from rnn_gru import RNNGRU
from rnn_lstm import RNNLSTM
from load_data import *

BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"We are using: {device}")

if __name__ == "__main__":
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
