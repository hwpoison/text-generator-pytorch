import torch
from utils import RedirectStdout
from model import RNNModule, predict
import numpy as np
import time, os

FILE_NAME = 'brain.pth'
model_state = torch.load(FILE_NAME)

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

process_data = model_state['process_data']
flags = model_state['flags']
n_vocab = process_data['n_vocab']
vocab_to_int = process_data['vocab_to_int']
int_to_vocab = process_data['int_to_vocab']

#initialize and load net state
net_state = model_state['model_state']
net = RNNModule(n_vocab, flags.seq_size, flags.embedding_size, flags.lstm_size)
net.load_state_dict(net_state)

redirect = RedirectStdout()
seq_len = 150
PROMPT_SYMBOL = ':'

while True:
    words = input(PROMPT_SYMBOL)
    try:
       output = predict(device, net, words.split(' '), 
               n_vocab, vocab_to_int, int_to_vocab, top_k=1, seq_size=seq_len)
       redirect.start()
       print(PROMPT_SYMBOL + words + ' ', end='')
       redirect.stop()
       for word in output[len(words):]:
            redirect.start()
            print(word, '', end='')
            #time.sleep(0.2)
            os.system("clear")
            redirect.stop()
    except KeyError:
        redirect.start()
        print("Error!, please try with other words")
        redirect.stop()
    input("\nPress a key for other input...")
    redirect.sclear()
