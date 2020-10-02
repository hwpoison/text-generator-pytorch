import sys
import os
import math
import time 
import numpy as np
from collections import Counter

def asMinutes(s): # get current time.time() to minutes
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent): # get time since start
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def get_data_from_file(train_file, batch_size, seq_size):
    with open(train_file, 'r') as f:
        text = f.read().lower()
    text = text.split()
    # random samples from text for test
    random_samples = []
    for sec in range(30):
        start_pos = np.random.randint(0, len(text))
        random_samples.append(text[start_pos:start_pos+10])
       
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)

    print('Vocabulary size in',train_file,':', n_vocab)

    int_text = [vocab_to_int[w] for w in text]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]

    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text, random_samples

def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]

def array_to_vocab(array, int_to_vocab):
        return ' '.join([int_to_vocab[w] for w in array])

class RedirectStdout():
    stdout = sys.stdout
    stdout_log = []

    def start(self):
        sys.stdout = self

    def stop(self):
        sys.stdout = self.stdout
        print("".join([i for i in self.stdout_log]))
        stdout_log = []

    def write(self, text):
        self.stdout_log.append(text)

    def flush(self):
        pass

    def sclear(self):
        if os.sys.platform == 'win32':
            os.system("cls")
        else:
            os.system("clear")
        self.stdout_log = []

if __name__ == '__main__':
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text, random_sentece = get_data_from_file('asimov.txt', 64,5)
    b = get_batches(in_text, out_text, 64,2)
    for x,y in b:
        print(array_to_vocab(x[0], int_to_vocab), '-', array_to_vocab(y[0], int_to_vocab))
