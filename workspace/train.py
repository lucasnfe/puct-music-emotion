import os
import time
import math
import json
import torch
import argparse

from encoder import *
from emopia import Emopia
from model import MusicGenerator

def save_model(model, optimizer, epoch, save_to):
    model_path = save_to.format(epoch)
    torch.save(
        dict(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch
        ),
        model_path)

def train(model, train_data, test_data, epochs, lr, save_to):
    best_model = None
    best_val_loss = float('inf')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # Train model for one epoch
        train_step(model, train_data, epoch, lr, criterion, optimizer, scheduler)

        # Evaluate model on test set
        val_loss = evaluate(model, test_data, criterion)

        elapsed = time.time() - epoch_start_time

        # Compute validation perplexity
        val_ppl = math.exp(val_loss)

        # Log training statistics for this epoch
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')

        # Save best model so far
        if val_loss < best_val_loss:
            print(f'Validation loss improved from {best_val_loss:5.2f} to {val_loss:5.2f}.')
            best_val_loss = val_loss
       
        print(f'Saving model to {save_to}.')
        save_model(model, optimizer, epoch, save_to)

        print('-' * 89)

        # Advance one epoch of the learning rate scheduler
        scheduler.step()

    return best_model

def train_step(model, train_data, epoch, lr, criterion, optimizer, scheduler, log_interval=100):
    model.train()
    start_time = time.time()

    total_loss = 0
    for batch, (x, y, lengths) in enumerate(train_data):
        # Forward pass
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.to(device)
        
        y_hat = model(x, lengths)

        # Backward pass
        optimizer.zero_grad()
        loss = criterion(y_hat.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()

        # Log training statistics
        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            log_stats(scheduler, epoch, batch, len(train_data), total_loss, start_time, log_interval)
            total_loss = 0
            start_time = time.time()

def log_stats(scheduler, epoch, batch, num_batches, total_loss, start_time, log_interval):
    # Get current learning rate
    lr = scheduler.get_last_lr()[0]

    # Compute duration of each batch
    ms_per_batch = (time.time() - start_time) * 1000 / log_interval

    # Compute current loss
    cur_loss = total_loss / log_interval

    # compute current perplexity
    ppl = math.exp(cur_loss)

    print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
          f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
          f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')

def evaluate(model, test_data, criterion):
    model.eval()

    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch, (x, y, lengths) in enumerate(test_data):
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)

            # Evaluate
            y_hat = model(x, lengths)
            loss = criterion(y_hat.view(-1, vocab_size), y.view(-1))

            total_loss += x.shape[0] * loss.item()
            total_samples += x.shape[0]

    return total_loss / total_samples

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='train_ftm.py')
    parser.add_argument('--train', type=str, required=True, help="Path to train data directory.")
    parser.add_argument('--test', type=str, required=True, help="Path to test data directory.")
    parser.add_argument('--model', type=str, default=None, help="Path to load model from.")
    parser.add_argument('--epochs', type=int, default=100, help="Epochs to train.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument('--seq_len', type=int, required=True, help="Max sequence to process.")
    parser.add_argument('--n_layers', type=int, default=8, help="Number of transformer layers.")
    parser.add_argument('--d_model', type=int, default=256, help="Dimension of the query matrix.")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--beat_resol', type=int, default=1024, help="Ticks per beat.")
    parser.add_argument('--save_to', type=str, required=True, help="Set a file to save the models to.")
    args = parser.parse_args()

    # Set up torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vocab_size = VOCAB_SIZE 

    # Get pad token
    pad_token = Event(event_type='control', value=3).to_int()

    # Load data as a flat tensors
    train_data = Emopia(args.train, pad_token=pad_token)
    test_data = Emopia(args.test, pad_token=pad_token)

    # Batchfy flat tensor data
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Build linear transformer
    model = MusicGenerator(n_tokens=vocab_size,
                            d_model=args.d_model,
                            seq_len=args.seq_len,
                     attention_type="causal-linear",
                           n_layers=args.n_layers,
                           n_heads=args.n_heads).to(device)


    # Load model
    if args.model:
        print('> Training from checkpoint', args.model)
        model.load_state_dict(torch.load(args.model, map_location=device)["model_state"])

    # Train model
    trained_model = train(model, train_dataloader, test_dataloader, epochs=args.epochs, lr=args.lr, save_to=args.save_to)
