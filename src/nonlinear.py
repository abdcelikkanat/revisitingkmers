import torch
import math
import time
import sys
import random
import pickle as pkl
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from functools import partial
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import re

def set_seed(seed):
    # Set the seed
    random.seed(seed)
    torch.manual_seed(seed)


class PairDataset(Dataset):
    def __init__(self, file_path, transform_func, neg_sample_per_pos=1000, max_read_num=0, verbose=True, seed=0):

        # Set the parameters
        self.__both_kmer_profiles = None
        self.__transform_func = transform_func
        self.__neg_sample_per_pos = neg_sample_per_pos
        self.__seed = seed

        # Set the seed
        set_seed(seed)

        # Get the number of lines
        with open(file_path, 'r') as f:
            lines_num = sum(1 for _ in f)
        # If the max_read_num is set, then sample the line number to read
        if max_read_num > 0:
            chosen_lines = random.sample(range(lines_num), max_read_num)
            chosen_lines.sort()

        # Read the file
        chosen_line_idx = 0
        left_kmer_profiles, right_kmer_profiles = [], []
        with open(file_path, 'r') as f:
            for current_line_idx, line in enumerate(f):

                if max_read_num > 0:
                    if chosen_line_idx == len(chosen_lines):
                        break

                    if current_line_idx != chosen_lines[chosen_line_idx]:
                        continue
                    else:
                        chosen_line_idx += 1

                # Remove the newline character and commas
                left_read, right_read = line.strip().split(',')
                left_kmer_profiles.append(self.__transform_func(left_read))
                right_kmer_profiles.append(self.__transform_func(right_read))

        # Combine the left and right k-mer profiles
        self.__both_kmer_profiles = torch.from_numpy(
            np.asarray(left_kmer_profiles + right_kmer_profiles)
        ).to(torch.float)

        if verbose:
            print(f"The data file was read successfully!")
            print(f"\t+ Total number of read pairs: {lines_num}")
            if max_read_num > 0:
                print(f"\t+ Number of read pairs used: {max_read_num}")

        # Temporary variables
        self.__ones = torch.ones((len(self.__both_kmer_profiles),))

    def __len__(self):

        return len(self.__both_kmer_profiles) // 2

    def __getitem__(self, idx):

        # Sample negative sample_indices
        negative_sample_indices = torch.multinomial(
            self.__ones, replacement=True, num_samples=2*self.__neg_sample_per_pos
        )

        # Define the positive and negative k-mer profile pairs
        left_kmer_profiles = torch.concatenate((
            self.__both_kmer_profiles[idx].unsqueeze(0),
            self.__both_kmer_profiles[negative_sample_indices[:self.__neg_sample_per_pos]]
        ))
        right_kmer_profiles = torch.concatenate((
            self.__both_kmer_profiles[idx+self.__len__()].unsqueeze(0),
            self.__both_kmer_profiles[negative_sample_indices[self.__neg_sample_per_pos:]]
        ))
        # Define the labels
        labels = torch.tensor([1] + [0] * self.__neg_sample_per_pos, dtype=torch.float)

        return left_kmer_profiles, right_kmer_profiles, labels


class NonLinearModel(torch.nn.Module):
    def __init__(self, k, dim=256, device=torch.device("cpu"), verbose=False, seed=0):
        super(NonLinearModel, self).__init__()

        # Set the parameters
        self.__device = device
        self.__verbose = verbose

        # Define the letters, k-mer size, and the base complement
        self.__k = k
        self.__dim = dim
        self.__letters = ['A', 'C', 'G', 'T']
        self.__kmer2id = {''.join(kmer): i for i, kmer in enumerate(itertools.product(self.__letters, repeat=self.__k))}
        self.__kmers_num = len(self.__kmer2id)

        # Set the seed
        set_seed(seed)

        # Define the layers
        self.linear1 = torch.nn.Linear(self.__kmers_num, 512, dtype=torch.float, device=self.__device)
        self.batch1 = torch.nn.BatchNorm1d(512, dtype=torch.float, device=self.__device)
        self.activation1 = torch.nn.Sigmoid()
        self.dropout1 = torch.nn.Dropout(0.2)
        self.linear2 = torch.nn.Linear(512, self.__dim, dtype=torch.float, device=self.__device)

        self.bce_loss = torch.nn.BCELoss()

    def encoder(self, kmer_profile):

        output = self.linear1(kmer_profile)
        output = self.batch1(output)
        output = self.activation1(output)
        output = self.dropout1(output)
        output = self.linear2(output)

        return output

    def forward(self, left_kmer_profile, right_kmer_profile):
        left_output = self.encoder(left_kmer_profile)
        right_output = self.encoder(right_kmer_profile)

        return left_output, right_output

    def get_k(self):
        return self.__k

    def get_dim(self):
        return self.__dim

    def get_device(self):
        return self.__device

    def read2kmer_profile(self, read, normalized=True):

        # Get the k-mer profile
        kmer2id = [self.__kmer2id[read[i:i + self.__k]] for i in range(len(read) - self.__k + 1)]
        kmer_profile = np.bincount(kmer2id, minlength=self.__kmers_num)

        if normalized:
            kmer_profile = kmer_profile / kmer_profile.sum()

        return kmer_profile

    def read2emb(self, reads, normalized=True):

        with torch.no_grad():
            kmer_profiles = []
            for read in reads:
                kmer_profiles.append(self.read2kmer_profile(read, normalized=normalized))

            kmer_profiles = torch.from_numpy(np.asarray(kmer_profiles)).to(torch.float)
            embs = self.encoder(kmer_profiles).detach().numpy()

        return embs

def loss_func(left_embeddings, right_embeddings, labels, name="bern"):

    if name == "bern":
        p = torch.exp(-torch.norm(left_embeddings - right_embeddings, p=2, dim=1)**2 )

        return torch.nn.functional.binary_cross_entropy(p, labels, reduction='mean')

    elif name == "poisson":

        log_lambda = -torch.norm(left_embeddings - right_embeddings, p=2, dim=1)**2

        return torch.mean(-(labels * log_lambda) + torch.exp(log_lambda))

    elif name == "hinge":

        d = torch.norm(left_embeddings - right_embeddings, p=2, dim=1)
        return torch.mean(labels * (d**2) + (1 - labels) * torch.nn.functional.relu(1 - d)**2)

    else:

        raise ValueError(f"Unknown loss function: {name}")


def single_epoch(model, loss_func, optimizer, training_loader, loss_name=bern):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epoch_loss = 0.
    for data in training_loader:
        left_kmer_profile, right_kmer_profile, labels = data

        # Zero your gradients since PyTorch accumulates gradients on subsequent backward passes.
        optimizer.zero_grad()
        left_kmer_profile = left_kmer_profile.reshape(-1, left_kmer_profile.shape[-1]).to(device)
        right_kmer_profile = right_kmer_profile.reshape(-1, right_kmer_profile.shape[-1]).to(device)
        labels = labels.reshape(-1).to(device)

        # Make predictions for the current epoch
        left_output, right_output = model(left_kmer_profile, right_kmer_profile)

        # Compute the loss and backpropagate
        batch_loss = loss_func(left_output, right_output, labels, name=loss_name)
        batch_loss.backward()

        # Update the model parameters
        optimizer.step()

        # Get the epoch loss for reporting
        epoch_loss += batch_loss.item()
        del batch_loss, left_kmer_profile, right_kmer_profile, labels, left_output, right_output
        torch.cuda.empty_cache()

    return epoch_loss / len(training_loader)


def run(model, learning_rate, epoch_num, loss_name="bern", model_save_path=None, loss_file_path=None, checkpoint=0, verbose=True):

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model.to(torch.device("cuda:0"))

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if loss_file_path is not None:
        writer = SummaryWriter(loss_file_path)
    if verbose:
        print("Training has just started.")

    for epoch in range(epoch_num):
        if verbose:
            print(f"\t+ Epoch {epoch + 1}.")

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = single_epoch(model, loss_func, optimizer, training_loader, loss_name)

        if verbose:
            print(f"Epoch {epoch + 1}, Training Loss: {avg_loss}")

        if loss_file_path is not None:
            writer.add_scalar('Loss/train', avg_loss, epoch + 1)
            writer.flush()

        if model_save_path is not None and checkpoint > 0 and (epoch + 1) % checkpoint == 0:

            # model_save_path contains the substring epoch=ID, so change the ID to the current epoch
            temp_model_save_path = re.sub('epoch.*_LR', f"epoch={epoch + 1}_LR", model_save_path)

            torch.save([{'k': model.get_k(), 'device': model.get_device()}, model.state_dict()], temp_model_save_path)
            if verbose:
                print(f"Model is saving.")
                print(f"\t- Target path: {temp_model_save_path}")

    writer.close()

    if model_save_path is not None:
        # If the model is a DataParallel object, then save the model.module
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        torch.save([{'k': model.get_k(), 'dim': model.get_dim(), 'device': model.get_device()}, model.state_dict()], model_save_path)

        if verbose:
            print(f"Model is saving.")
            print(f"\t- Target path: {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clustering')
    parser.add_argument('--input', type=str, help='Input sequence file')
    parser.add_argument('--k', type=int, default=2, help='k value')
    parser.add_argument('--dim', type=int, default=256, help='dimension value')
    parser.add_argument('--neg_sample_per_pos', type=int, default=1000, help='Negative sample ratio')
    parser.add_argument('--max_read_num', type=int, default=10000, help='Maximum number of reads to get from the file')
    parser.add_argument('--epoch', type=int, default=1000, help='Epoch number')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=0, help='Batch size (0: no batch)')
    parser.add_argument('--device', type=str, default="cpu", help='Device (cpu or cuda)')
    parser.add_argument('--workers_num', type=int, default=1, help='Number of workers for data loader')
    parser.add_argument('--loss_name', type=str, default="bern", help='Loss function (bern, poisson, hinge)')
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--seed', type=int, default=26042024, help='Seed for random number generator')
    parser.add_argument('--checkpoint', type=int, default=0, help='Save the model for every checkpoint epoch')
    args = parser.parse_args()

    # Define the model
    model = NonLinearModel(
        k=args.k, dim=args.dim, device=torch.device(args.device), verbose=True, seed=args.seed
    )

    # Read the dataset
    training_dataset = PairDataset(
        file_path=args.input, transform_func=model.read2kmer_profile, neg_sample_per_pos=args.neg_sample_per_pos,
        max_read_num=args.max_read_num, seed=args.seed
    )
    # Define the training data loader
    training_loader = DataLoader(
        training_dataset, batch_size=args.batch_size if args.batch_size else len(training_dataset),
        shuffle=True, num_workers=args.workers_num
    )

    # Run the model
    run(model, args.lr, args.epoch, args.loss_name, args.output, args.output + ".loss", args.checkpoint, verbose=True)
