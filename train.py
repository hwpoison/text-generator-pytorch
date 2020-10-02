import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RNNModule, predict
import time
from argparse import Namespace
from utils import  asMinutes, timeSince, get_data_from_file, get_batches, array_to_vocab
from numpy.random import choice, randint 

flags = Namespace(
    train_file='asimov.txt',
    seq_size=16, 
    batch_size=64,
    embedding_size=64,
    lstm_size=64,
    gradients_norm=5,
    predict_top_k=5,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
int_to_vocab, vocab_to_int, n_vocab, in_text, out_text, random_samples = get_data_from_file( flags.train_file, flags.batch_size, flags.seq_size)

net = RNNModule(n_vocab, flags.seq_size,
                flags.embedding_size, flags.lstm_size)
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

def train():
    print("Training...")
    iteration = 0
    start = time.time()
    epochs = 100
    for e in range(1, epochs):
        batches = get_batches(in_text, out_text, flags.batch_size, flags.seq_size)
        state_h, state_c = net.zero_state(flags.batch_size)
        # Transfer data to GPU
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        for x, y in batches:
            iteration += 1
            
            # Tell it we are in training mode
            net.train()

            # Reset all gradients
            optimizer.zero_grad()

            # Transfer data to GPU
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)
            logits, (state_h, state_c) = net(x, (state_h, state_c))
            loss = criterion(logits.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss_value = loss.item()

            # Perform back-propagation
            loss.backward()

            # Gradient clip
            _ = torch.nn.utils.clip_grad_norm_(
                    net.parameters(), flags.gradients_norm)

            # Update the network's parameters
            optimizer.step()
            if iteration % 100 == 0:
                    selected_sample = random_samples[randint(len(random_samples))]
                    output = predict(device,net , selected_sample[:3],
                                  n_vocab, vocab_to_int, int_to_vocab, top_k=1)
                    # prediction preview
                    print('Input: ', ' '.join([w for w in selected_sample[:10]]))
                    print('Output:',   ' '.join([w for w in output[:10]]))
                    # show loss 
                    print(f'Epoch: {e}/{epochs} Iteration: {iteration} Loss: {loss_value:.4f}')
                    print(f'Current time:{timeSince(start, e/epochs)}   Total: { (e/epochs)*100:.1f}%')
                                                  
                    print("="*50)

                    data = {
                        'model_state':net.state_dict(),
                        'flags':flags,
                        'final_loss':loss_value,
                        'trained_in':str(timeSince(start, e/200)),
                        'process_data':{
                            'n_vocab':n_vocab,
                            'vocab_to_int':vocab_to_int,
                            'int_to_vocab':int_to_vocab
                         }
                    }
                    torch.save(data, f'brain.pth')
                    print('Model saved.')
    print("Trained.")

if __name__ == '__main__':
    train()
