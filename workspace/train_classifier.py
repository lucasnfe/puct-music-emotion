import torch
import time
import argparse
import numpy as np

from vgmidi import VGMidiLabelled
from sklearn.metrics import confusion_matrix
from model import MusicClassifier

from encoder import *

# Reproducitility
np.random.seed(42)
torch.manual_seed(42)

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
    model.train()

    best_model = None
    best_val_accuracy = 0

    if args.out_size == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # Train model for one epoch
        train_step(model, train_data, epoch, lr, criterion, optimizer, scheduler)

        # Evaluate model on test set
        val_accuracy, confusion = evaluate(model, test_data)

        elapsed = time.time() - epoch_start_time

        # Log training statistics for this epoch
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'accuracy {val_accuracy:5.2f} | best val accuracy: {best_val_accuracy:5.2f} ')

        # Save best model so far
        if val_accuracy > best_val_accuracy:
            print(f'Validation accuracy improved from {best_val_accuracy:5.2f} to {val_accuracy:5.2f}.'
                  f'Saving model to {save_to}.')

            print(confusion)

            best_val_accuracy = val_accuracy
            save_model(model, optimizer, epoch, save_to)

        print('-' * 89)
        scheduler.step()

def train_step(model, train_data, epoch, lr, criterion, optimizer, scheduler, log_interval=10):
    model.train()
    start_time = time.time()

    total_loss = 0
    for batch, (x, y) in enumerate(train_data):
        # Forward pass
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)

        # Backward pass
        optimizer.zero_grad()
        
        if args.out_size == 1:
            loss = criterion(y_hat.view(-1), y.view(-1))
        else:
            loss = criterion(y_hat.view(-1, args.out_size), y.view(-1))
        
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()

        # print statistics
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

    print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
          f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
          f'loss {cur_loss:5.5f}')

def evaluate(model, test_data):
    model.eval()

    ys = []
    ys_hat = []
    with torch.no_grad():
        for batch, (x, y) in enumerate(test_data):
            x = x.to(device)
            y = y.to(device)

            # Evaluate
            y_hat = model(x)

            # the class with the highest energy is what we choose as prediction
            if args.out_size == 1:
                y_hat = torch.round(torch.sigmoid(y_hat)).squeeze()
            else:
                _, y_hat = torch.max(y_hat.view(-1, args.out_size).data, dim=1)
            
            ys += y.squeeze().tolist()
            ys_hat += y_hat.tolist()

    print(ys)
    print(ys_hat)

    accuracy = np.mean(np.array(ys) == np.array(ys_hat))
    confusion = confusion_matrix(ys, ys_hat)

    return accuracy, confusion

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='train_emotion_classifier.py')
    parser.add_argument('--train', type=str, required=True, help="Path to train data directory.")
    parser.add_argument('--test', type=str, required=True, help="Path to train data directory.")
    parser.add_argument('--model', type=str, required=False, help="Path to load model from.")
    parser.add_argument('--epochs', type=int, default=100, help="Epochs to train.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument('--seq_len', type=int, required=True, help="Max sequence to process.")
    parser.add_argument('--n_layers', type=int, default=8, help="Number of transformer layers.")
    parser.add_argument('--d_model', type=int, default=512, help="Dimension of the query matrix.")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--save_to', type=str, required=True, help="Set a file to save the models to.")
    parser.add_argument('--out_size', type=int, required=True, help="Output size.")
    args = parser.parse_args()

    # Set up torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get pad token
    vocab_size = VOCAB_SIZE 

    # Load data as a flat tensors
    vgmidi_train = VGMidiLabelled(args.train)
    vgmidi_test = VGMidiLabelled(args.test)

    # Batchfy flat tensor data
    train_loader = torch.utils.data.DataLoader(vgmidi_train, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(vgmidi_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Build linear transformer
    model = MusicClassifier(n_tokens=vocab_size,
                            d_model=args.d_model,
                            seq_len=args.seq_len,
                     attention_type="linear",
                           n_layers=args.n_layers,
                           n_heads=args.n_heads).to(device)

    # Load model
    if args.model:
        print(f'> Fine-tuning model {args.model}')
        model.load_state_dict(torch.load(args.model, map_location=device)["model_state"])
   
    # Lock paramters and reset last l
    for param in model.pos_embedding.parameters():
        param.requires_grad = False
    
    for param in model.value_embedding.parameters():
        param.requires_grad = False
    
    for i, layer in enumerate(model.transformer.layers):
        if i < args.n_layers - 0:
            for param in layer.parameters():
                param.requires_grad = False

    for param in model.predictor.parameters():
        param.requires_grad = True
    
    # Add classification head
    model = torch.nn.Sequential(model,
                                torch.nn.Dropout(0.0),
                                torch.nn.Linear(vocab_size, args.out_size)).to(device)

    train(model, train_loader, test_loader, args.epochs, args.lr, args.save_to)
