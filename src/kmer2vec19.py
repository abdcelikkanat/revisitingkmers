import torch
import math
import time
import sys
import random
import pickle as pkl
import itertools
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process, Pool, Array, Manager
from functools import partial
import argparse


class KMer2Emb(torch.nn.Module):

    def __init__(self,  k, dim=2, lr=0.1, epoch_num=100, batch_size = 1000,
                 device=torch.device("cpu"), verbose=False, seed=0):

        super(KMer2Emb, self).__init__()

        self.__seed = seed
        self.__k = k
        self.__dim = dim
        self.__lr = lr
        self.__epoch_num = epoch_num
        self.__batch_size = batch_size
        self.__device = device
        self.__verbose = verbose

        self.__set_seed(seed)

        self.__letters = ['A', 'C', 'G', 'T']
        self.__kmer2id = {''.join(kmer): idx for idx, kmer in enumerate(itertools.product(self.__letters, repeat=k))}

        self.__embs = torch.nn.Parameter(
            2 * torch.rand(size=(4**self.__k, self.__dim), device=self.__device) - 1, requires_grad=True
        )

        self.__optimizer = torch.optim.Adam(self.parameters(), lr=self.__lr)
        self.__loss = []

    def set_device(self, device):

        self.__device = device
        self.__embs = self.__embs.to(device)

    def __set_seed(self, seed=None):

        if seed is not None:
            self._seed = seed

        random.seed(self._seed)
        torch.manual_seed(self._seed)
    def get_cooccurrence_counts(self, file_path, window_size, read_sample_size):

        # Get the number of lines in the file
        with open(file_path, 'r') as f:
            num_lines = sum(1 for _ in f)

        if self.__verbose:
            print(f"The input file has {num_lines} sequences.")

        # Initialize the counts
        counts = np.zeros(shape=(4 ** self.__k, 4 ** self.__k), dtype=float)

        if read_sample_size <= 0:
            read_sample_size = num_lines

        # Sample 'read_sample_size' lines and sort them
        indices = random.sample(range(num_lines), read_sample_size)
        indices.sort()

        # Read the sampled lines
        with open(file_path, 'r') as f:
            current_id_pos = 0
            line_id = 0
            for line in f:
                # Remove the newline character and commas
                left_read, right_read = line.strip().split(',')

                if line_id == indices[current_id_pos]:

                    for read in [left_read, right_read]:
                        # Get the center
                        for i in range(len(read) - self.__k + 1):
                            center_kmer_id = self.__kmer2id[read[i:i+self.__k]]
                            # Get the context
                            for j in range(max(0, i - window_size), min(len(read) - self.__k + 1, i + window_size + 1)):
                                if j == i:
                                    continue
                                context_kmer_id = self.__kmer2id[read[j:j+self.__k]]

                                if center_kmer_id <= context_kmer_id:
                                    # Update the counts
                                    counts[center_kmer_id, context_kmer_id] += 1

                    # Increment the current_id_pos to go to the next selected line
                    current_id_pos += 1
                    if current_id_pos == len(indices):
                        break

                line_id += 1

        # Add the upper triangular matrix to the lower triangular matrix
        return counts[np.triu_indices(4**self.__k, k=1)] / (2*read_sample_size)

    def __compute_loss(self, pairs, counts):

        dist = torch.norm(
            torch.index_select(self.__embs, 0, pairs[0]) - torch.index_select(self.__embs, 0, pairs[1]),
            p=1, dim=1
        )
        log_rate = - dist

        return -(counts * log_rate - torch.exp(log_rate)).sum()

    def learn(self, file_path, window_size, read_sample_size=10000):

        pairs = torch.triu_indices(4 ** self.__k, 4 ** self.__k, offset=1, device=self.__device)
        counts = torch.from_numpy(
            self.get_cooccurrence_counts(file_path, window_size, read_sample_size)
        ).to(self.__device)

        for epoch in range(self.__epoch_num):

            # Shuffle data
            indices = torch.randperm(pairs.shape[1])
            pairs = pairs[:, indices]
            counts = counts[indices]

            epoch_loss = 0
            batch_size = self.__batch_size if self.__batch_size > 0 else pairs.shape[1]
            for i in range(0, pairs.shape[1], batch_size):

                batch_pairs = pairs[:, i:i+batch_size]
                batch_counts = counts[i:i+batch_size]

                if batch_pairs.shape[1] != batch_size:
                    continue

                self.__optimizer.zero_grad()

                batch_loss = self.__compute_loss(batch_pairs, batch_counts)
                batch_loss.backward()
                self.__optimizer.step()

                self.__optimizer.zero_grad()

                epoch_loss += batch_loss.item()

            epoch_loss /= math.ceil(pairs.shape[1] / batch_size)

            if self.__verbose:
                print(f"epoch: {epoch}, loss: {epoch_loss}")

            self.__loss.append(epoch_loss)

        return self.__loss

    def save(self, file_path):

        if self.__verbose:
            print(f"+ Model file is saving.")
            print(f"\t- Target path: {file_path}")

        kwargs = {
            'k': self.__k,
            'dim': self.__dim,
            'lr': self.__lr,
            'epoch_num': self.__epoch_num,
            'batch_size': self.__batch_size,
            'device': self.__device,
            'verbose': self.__verbose,
            'seed': self.__seed
        }

        torch.save([kwargs, self.state_dict()], file_path)

    def get_emb(self, sequences):

        embeddings = []
        for sequence in sequences:

            # Get the k-mer profile
            kmer_profile = torch.zeros(size=(4 ** self.__k, ), device=self.__device, dtype=torch.float)
            for i in range(len(sequence) - self.__k + 1):
                kmer_id = self.__kmer2id[sequence[i:i+self.__k]]
                kmer_profile[kmer_id] += 1

            # Normalize the k-mer profile
            kmer_profile = kmer_profile / torch.norm(kmer_profile, p=1)

            emb = kmer_profile @ self.__embs
            # emb = emb / (len(sequence) - self.__k + 1)
            embeddings.append(emb.detach().numpy())

        return np.asarray(embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clustering')
    parser.add_argument('--input', type=str, help='Input sequence file')
    parser.add_argument('--k', type=int, default=2, help='k value')
    parser.add_argument('--dim', type=int, default=2, help='Dimension of k-mer embedding')
    parser.add_argument('--epoch', type=int, default=1000, help='Epoch number')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=0, help='Batch size (0: no batch)')
    parser.add_argument('--device', type=str, default="cpu", help='Device (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=26042024, help='Seed for random number generator')
    parser.add_argument('--w', type=int, default=2, help='Window size')
    parser.add_argument('--read_sample_size', type=int, default=100000, help='Read sample size')
    parser.add_argument('--output', type=str, help='Output file')
    args = parser.parse_args()

    kmer2emb = KMer2Emb(
        k=args.k, dim=args.dim,
        lr=args.lr, epoch_num=args.epoch, batch_size=args.batch_size,
        device=torch.device(args.device), verbose=True, seed=args.seed
    )
    loss = kmer2emb.learn(file_path=args.input, window_size=args.w, read_sample_size=args.read_sample_size)

    # Save the model
    kmer2emb.save(args.output)

    # Save the loss
    with open(args.output + ".loss", 'w') as f:
        for l in loss:
            f.write(f"{l}\n")
    ''' '''

    # plt.figure()
    # plt.plot(loss)
    # plt.show()

    # kwargs, model_state_dict = torch.load(args.output)
    # new_model = KMer2Emb(**kwargs)
    # new_model.load_state_dict(model_state_dict)
    # print(new_model.get_emb(["AACCGT", "AA"]))

    # saved_model = torch.load("./deneme.model")
    # saved_model = KMer2Emb()
    # saved_model.load_state_dict(torch.load('./deneme.model'))
    # # Get the embedding of the first sequence
    # with open(args.input, 'r') as f:
    #     sequence = f.readline().strip().replace(',', '')
    # emb = saved_model.get_emb(sequence)
    # print(emb)
    # emb = kmer2emb.get_emb(sequence)
    # print(emb)
    ''' 
    kwargs, model_state_dict = torch.load(args.output)
    new_model = KMer2Emb(**kwargs)
    new_model.load_state_dict(model_state_dict)

    sequences = []
    with open(args.input, 'r') as f:
        for line in f:
            sequence = line.strip().replace(',', '')
            sequences.append(sequence)

    embs = new_model.get_emb(sequences)

    plt.figure()
    for idx, start_idx in enumerate(range(0, len(embs), 1000)):
        plt.scatter(embs[start_idx:start_idx+1000, 0], embs[start_idx:start_idx+1000, 1], s=1)
    # plt.scatter(embs[:, 0], embs[:, 1], s=1, c=colors[idx])
    plt.show()
    '''
