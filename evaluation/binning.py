# This script has been adapted from the file available at the following address:
# https://github.com/MAGICS-LAB/DNABERT_S/blob/main/evaluate/eval_binning.py
import csv
import argparse
import os
import sys
import collections
import numpy as np
import sklearn.metrics
from datetime import datetime
from evaluation.utils import align_labels_via_hungarian_algorithm
from evaluation.utils import get_embedding, KMedoid, compute_class_center_medium_similarity

csv.field_size_limit(sys.maxsize)
csv.field_size_limit(sys.maxsize)
MAX_SEQ_LEN = 20000
MIN_SEQ_LEN = 2500
MIN_ABUNDANCE_VALUE = 10


def main(args):
    model_list = args.model_list.split(",")
    for model_name in model_list:
        for species in args.species.split(","):
            for sample in map(int, args.samples.split(",")):

                # Define the appropriate metric for the given method
                if args.metric != None:
                    metric = args.metric
                else:
                    if model_name == "kmerprofile":
                        metric = "l1"
                    elif model_name == "nonlinear":
                        metric = "l2"
                    else:
                        metric = "dot"

                print(f"Model: {model_name} Species: {species} Sample ID: {sample} Metric: {metric}")

                # Load the clustering data to compute similarity threshold
                clustering_data_file_path = os.path.join(args.data_dir, species, f"clustering_0.tsv")
                with open(clustering_data_file_path, "r") as f:
                    reader = csv.reader(f, delimiter="\t")
                    data = list(reader)[1:]

                # Shorten the sequences if they are longer than the
                dna_sequences = [d[0][:MAX_SEQ_LEN] for d in data]
                labels = [d[1] for d in data]

                # convert labels to numeric values
                label2id = {l: i for i, l in enumerate(set(labels))}
                labels = np.array([label2id[l] for l in labels])
                num_clusters = len(label2id)
                print(f"Clustering data contains {len(dna_sequences)} sequences with {num_clusters} clusters.")

                # Get embeddings
                embedding = get_embedding(
                    dna_sequences=dna_sequences, model_name=model_name, species=species, sample=0,
                    k=args.k, task_name="clustering", test_model_dir=args.test_model_dir
                )

                percentile_values = compute_class_center_medium_similarity(embedding, labels, metric=metric)
                threshold = percentile_values[-3]
                print(f"Threshold value: {threshold}")

                # Load binning data
                data_file = os.path.join(args.data_dir, species, f"binning_{sample}.tsv")

                with open(data_file, "r") as f:
                    reader = csv.reader(f, delimiter="\t")
                    data = list(reader)[1:]

                dna_sequences = [d[0][:MAX_SEQ_LEN] for d in data]
                labels_bin = [d[1] for d in data]

                # filter sequences with length < 2500
                filterd_idx = [i for i, seq in enumerate(dna_sequences) if len(seq) >= MIN_SEQ_LEN]
                dna_sequences = [dna_sequences[i] for i in filterd_idx]
                labels_bin = [labels_bin[i] for i in filterd_idx]

                # filter sequences with low abundance labels (less than 10)
                label_counts = collections.Counter(labels_bin)
                filterd_idx = [i for i, l in enumerate(labels_bin) if label_counts[l] >= MIN_ABUNDANCE_VALUE]
                dna_sequences = [dna_sequences[i] for i in filterd_idx]
                labels_bin = [labels_bin[i] for i in filterd_idx]

                # convert labels to numeric values
                label2id = {l: i for i, l in enumerate(set(labels_bin))}
                labels_bin = np.array([label2id[l] for l in labels_bin])
                num_clusters = len(label2id)
                print(f"Get {len(dna_sequences)} sequences, {num_clusters} clusters")

                # Generate embeddings for the binning set
                embedding = get_embedding(
                    dna_sequences, model_name, species, sample, k=args.k, metric=metric,
                    task_name="binning", test_model_dir=args.test_model_dir
                )

                # Run the KMedoid algorithm
                binning_results = KMedoid(
                    embedding, min_similarity=threshold, min_bin_size=10,
                    max_iter=1000, metric=metric, scalable=args.scalable
                )

                # Get the number of true labels and predictied labels
                true_labels_bin = labels_bin[binning_results != -1]
                predicted_labels = binning_results[binning_results != -1]
                print("Number of predicted labels: ", len(predicted_labels))

                # Align labels
                alignment_bin = align_labels_via_hungarian_algorithm(true_labels_bin, predicted_labels)
                predicted_labels_bin = [alignment_bin[label] for label in predicted_labels]

                # Calculate purity, completeness, recall, and ARI
                recall_bin = sklearn.metrics.recall_score(
                    true_labels_bin, predicted_labels_bin, average=None, zero_division=0
                )
                recall_bin.sort()

                f1_bin = sklearn.metrics.f1_score(true_labels_bin, predicted_labels_bin, average=None, zero_division=0)
                f1_bin.sort()
                thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                recall_results = []
                f1_results = []
                for threshold in thresholds:
                    recall_results.append(len(np.where(recall_bin > threshold)[0]))
                    f1_results.append(len(np.where(f1_bin > threshold)[0]))

                print(f"f1_results: {f1_results}")
                print(f"recall_results: {recall_results} \n")

                with open(args.output, 'a+') as f:
                    f.write("\n")
                    f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                    f.write(f"model: {model_name}, species: {species}, sample: {sample}, binning\n")
                    f.write(f"recall_results: {recall_results}\n")
                    f.write(f"f1_results: {f1_results}\n")
                    f.write(f"threshold: {threshold}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clustering')
    parser.add_argument(
        '--species', type=str, default="reference,marine,plant", help='Species to evaluate'
    )
    parser.add_argument(
        '--samples', type=str, default="5,6",
        help='Species to evaluate'
    )
    parser.add_argument(
        '--output', type=str,
        help='Output file path'
    )
    parser.add_argument(
        '--test_model_dir', type=str, default="/root/trained_model",
        help='Directory to save trained models to test'
    )
    parser.add_argument(
        '--model_list', type=str, default="dnaberts",
        help='List of models to evaluate, separated by comma. Currently support [tnf, tnf-k, dnabert2, hyenadna, nt, dnarberts, kmerprofile, poisson, nonlinear]'
    )
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument(
        '--k', type=int, default=4,
        help="k Value for the kmerprofile method"
    )
    parser.add_argument(
        '--metric', type=str, default=None,
        help="Metric to measure the similarities among embeddings"
    )
    parser.add_argument(
        '--scalable', type=bool, default=0,
        help="Controls how we compute the similarity among embeddings"
    )
    args = parser.parse_args()
    main(args)
