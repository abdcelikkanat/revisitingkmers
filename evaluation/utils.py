# This script has been adapted from the file available at the following address:
# https://github.com/MAGICS-LAB/DNABERT_S/blob/main/evaluate/eval_binning.py
import numpy as np
import transformers
import torch
import torch.utils.data as util_data
import torch.nn as nn
import tqdm
import os
from scipy.spatial import distance
import itertools
from sklearn.preprocessing import normalize
from src.poisson_model import PoissonModel
from src.nonlinear import NonLinearModel
from scipy.optimize import linear_sum_assignment

MODEL2BATCH_SIZE = {
    "tnf": 100, "tnf_k": 100, "hyenadna": 100, "dnabert2": 20, "nt": 64, "dnaberts": 20,
    "kmerprofile": -1, "poisson": -1, "nonlinear": -1,
}


def get_embedding(
        dna_sequences, model_name, species, sample, k=4,
        metric=None, task_name="clustering", test_model_dir=None, suffix=""
):
    #
    # model2batch_size = {
    #     "tnf": 100, "tnf_k": 100, "dna2vec": 100, "hyenadna": 100, "dnabert2": 20, "nt": 64, "dnaberts": 20,
    #     "nonlinear": -1, "poisson_model": -1, "linear": -1
    # }
    batch_size = 10  # MODEL2BATCH_SIZE[model_name]

    # Define the embedding directory and path
    embedding_file_dir = os.path.join("embeddings", species, f"{task_name}_{sample}{suffix}")
    embedding_file_path = os.path.join(embedding_file_dir, f"{model_name}.npy")

    # Load the embedding file if it exits
    if os.path.exists(embedding_file_path):
        print(f"Load embedding from file {embedding_file_path}")
        embedding = np.load(embedding_file_path)

    else:
        print(f"Calculate embedding for {model_name} {species} {sample}")

        if model_name == "tnf":

            embedding = calculate_tnf(dna_sequences)
            embedding = normalize(embedding, norm='l2')

        elif model_name == "tnf_k":

            embedding = calculate_tnf(dna_sequences, kernel=True)
            embedding = normalize(embedding, norm='l2')

        elif model_name == "hyenadna":

            embedding = calculate_llm_embedding(
                dna_sequences,
                model_name_or_path="LongSafari/hyenadna-medium-450k-seqlen-hf",
                model_max_length=20000,
                batch_size=batch_size
            )
            embedding = normalize(embedding, norm='l2')

        elif model_name == "dnabert2":

            embedding = calculate_llm_embedding(
                dna_sequences,
                model_name_or_path="zhihan1996/DNABERT-2-117M",
                model_max_length=5000,
                batch_size=batch_size
            )
            embedding = normalize(embedding, norm='l2')

        elif model_name == "nt":

            embedding = calculate_llm_embedding(
                dna_sequences,
                model_name_or_path="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
                model_max_length=2048,
                batch_size=batch_size
            )
            embedding = normalize(embedding, norm='l2')

        elif model_name == "dnaberts":

            embedding = calculate_llm_embedding(
                dna_sequences,
                model_name_or_path=test_model_dir,
                model_max_length=5000,
                batch_size=batch_size
            )
            embedding = normalize(embedding, norm='l2')

        elif model_name == "kmerprofile":

            norm = 'l1' if metric is None else metric
            embedding = calculate_tnf(dna_sequences, k=k)
            embedding = normalize(embedding, norm=norm)

        elif model_name == "poisson":

            kwargs, model_state_dict = torch.load(test_model_dir, map_location=torch.device("cpu"))
            kwargs['device'] = "cpu"
            pm = PoissonModel(**kwargs)
            pm.load_state_dict(model_state_dict)
            embedding = pm.read2emb(dna_sequences)

        elif model_name == "nonlinear":

            kwargs, model_state_dict = torch.load(test_model_dir, map_location=torch.device("cpu"))
            kwargs['device'] = "cpu"
            nlm = NonLinearModel(**kwargs)
            nlm.load_state_dict(model_state_dict)
            embedding = nlm.read2emb(dna_sequences)

        else:
            raise ValueError(f"Unknown model {model_name}")

        # Save the embedding file
        os.makedirs(embedding_file_dir, exist_ok=True)
        with open(embedding_file_path, 'wb') as f:
            np.save(f, embedding)

    return embedding


def calculate_tnf(dna_sequences, kernel=False, k=4):
    # Define all possible tetra-nucleotides
    nucleotides = ['A', 'T', 'C', 'G']

    multi_nucleotides = [''.join(kmer) for kmer in itertools.product(nucleotides, repeat=k)]

    # build mapping from multi-nucleotide to index
    tnf_index = {tn: i for i, tn in enumerate(multi_nucleotides)}

    # Iterate over each sequence and update counts
    embedding = np.zeros((len(dna_sequences), len(multi_nucleotides)))
    for j, seq in enumerate(dna_sequences):
        for i in range(len(seq) - k + 1):
            multi_nuc = seq[i:i + k]
            embedding[j, tnf_index[multi_nuc]] += 1

    if kernel:
        raise ValueError("Not Implemented!")

    return embedding


def calculate_llm_embedding(dna_sequences, model_name_or_path, model_max_length=400, batch_size=20):

    # reorder the sequences by length
    lengths = [len(seq) for seq in dna_sequences]
    idx = np.argsort(lengths)
    dna_sequences = [dna_sequences[i] for i in idx]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    is_hyenadna = "hyenadna" in model_name_or_path
    is_nt = "nucleotide-transformer" in model_name_or_path

    if is_nt:
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    else:
        model = transformers.AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

    n_gpu = torch.cuda.device_count()
    if n_gpu >= 1:
        model = nn.DataParallel(model)
        model.to("cuda")
        n_cpu = 0
    else:
        model.to("cpu")
        n_cpu = 1

    train_loader = util_data.DataLoader(
        dna_sequences, batch_size=batch_size * (n_gpu + n_cpu), shuffle=False, num_workers=2 * (n_gpu + n_cpu)
    )
    for j, batch in enumerate(tqdm.tqdm(train_loader)):
        with torch.no_grad():
            token_feat = tokenizer.batch_encode_plus(
                batch,
                max_length=model_max_length,
                return_tensors='pt',
                padding='longest',
                truncation=True
            )
            input_ids = token_feat['input_ids']
            if not is_hyenadna:
                attention_mask = token_feat['attention_mask']
            if n_gpu:
                input_ids = input_ids.cuda()
                if not is_hyenadna:
                    attention_mask = attention_mask.cuda()

            if is_hyenadna:
                model_output = model.forward(input_ids=input_ids)[0].detach().cpu()
                attention_mask = torch.ones(size=(model_output.shape[0], model_output.shape[1], 1), device='cpu')
            else:
                model_output = model.forward(input_ids=input_ids, attention_mask=attention_mask)[0].detach().cpu()
                attention_mask = attention_mask.unsqueeze(-1).detach().cpu()

            embedding = torch.sum(model_output * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)

            if j == 0:
                embeddings = embedding
            else:

                embeddings = torch.cat((embeddings, embedding), dim=0)

    embeddings = np.array(embeddings.detach().cpu())

    # reorder the embeddings
    embeddings = embeddings[np.argsort(idx)]

    return embeddings


def KMedoid(features, min_similarity=0.8, min_bin_size=100, max_iter=300, metric="dot", scalable=False):
    # rank nodes by the number of neighbors
    features = features.astype(np.float32)
    if metric == "dot":

        if scalable:
            similarities = np.zeros((features.shape[0], features.shape[0]))
            for i in range(0, features.shape[0], 512):
                similarities[i:i + 512, :] = features[i:i + 512, :] @ features.T
        else:
            similarities = np.dot(features, features.T)

    elif metric == "euclidean" or metric == "l2":
        similarities = np.exp(-distance.squareform(distance.pdist(features, 'euclidean')))
    elif metric == "l1":
        similarities = np.exp(-distance.squareform(distance.pdist(features, 'minkowski', p=1.)))
    else:
        raise ValueError("Invalid metric!")

    # set the values below min_similarity to 0
    similarities[similarities < min_similarity] = 0

    p = -np.ones(len(features), dtype=int)
    row_sum = np.sum(similarities, axis=1)

    iter_count = 1
    while np.any(p == -1):

        if iter_count == max_iter:
            break

        # Select the seed index, i.e. medoid index (Line 4)
        s = np.argmax(row_sum)
        # Initialize the current medoid (Line 4)
        current_medoid = features[s]
        selected_idx = None
        # Optimize the current medoid (Line 5-8)
        for t in range(3):
            # For the current medoid, find its similarities
            if metric == "dot":
                similarity = np.dot(features, current_medoid)
            elif metric == "euclidean" or metric == "l2":
                similarity = np.exp(
                    -distance.cdist(features, np.expand_dims(current_medoid, axis=0), 'euclidean')
                ).squeeze()
            elif metric == "l1":
                similarity = np.exp(
                    -distance.cdist(features, np.expand_dims(current_medoid, axis=0), 'minkowski', p=1.)
                ).squeeze()
            else:
                raise ValueError("Invalid metric!")
            # Determine the indices that are within the similarity threshold
            idx_within = similarity >= min_similarity
            # Determine the available indices, i.e. the indices that have not been assigned to a cluster yet
            idx_available = p == -1
            # Get the indices that are both within the similarity threshold and available
            selected_idx = np.where(np.logical_and(idx_within, idx_available))[0]
            # Determine the new medoid
            current_medoid = np.mean(features[selected_idx], axis=0)

        # Assign the cluster labels and update the row sums (Lines 9-10)
        if selected_idx is not None:
            p[selected_idx] = iter_count
            row_sum -= np.sum(similarities[:, selected_idx], axis=1)
            row_sum[selected_idx] = 0
            print(f"Current label: {iter_count}, Number of assigned elements: {len(selected_idx)}")
        else:
            raise ValueError("No selected index")

        iter_count += 1

    # remove bins that are too small
    unique, counts = np.unique(p, return_counts=True)
    for label, count in zip(unique, counts):
        if count < min_bin_size:
            p[p == label] = -1

    return p


def align_labels_via_hungarian_algorithm(true_labels, predicted_labels):
    """
    Aligns the predicted labels with the true labels using the Hungarian algorithm.

    Args:
    true_labels (list or array): The true labels of the data.
    predicted_labels (list or array): The labels predicted by a clustering algorithm.

    Returns:
    dict: A dictionary mapping the predicted labels to the aligned true labels.
    """
    # Create a confusion matrix
    max_label = max(max(true_labels), max(predicted_labels)) + 1
    confusion_matrix = np.zeros((max_label, max_label), dtype=int)

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[true_label, predicted_label] += 1

    # Apply the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(confusion_matrix, maximize=True)

    # Create a mapping from predicted labels to true labels
    label_mapping = {predicted_label: true_label for true_label, predicted_label in zip(row_ind, col_ind)}

    return label_mapping


def compute_class_center_medium_similarity(embeddings, labels, metric="dot"):
    idx = np.argsort(labels)
    embeddings = embeddings[idx]
    labels = labels[idx]
    n_sample_per_class = np.bincount(labels)

    all_similarities = np.zeros(len(embeddings))
    count = 0

    for i in range(len(n_sample_per_class)):
        start = count
        end = count + n_sample_per_class[i]
        mean = np.mean(embeddings[start:end], axis=0)
        if metric == "dot":
            similarities = np.dot(mean, embeddings[start:end].T).reshape(-1)
        elif metric == "euclidean" or metric == "l2":
            similarities = np.exp(
                -distance.cdist(np.expand_dims(mean, axis=0), embeddings[start:end], 'minkowski', p=2.).reshape(-1)
            )
        elif metric == "l1":
            similarities = np.exp(
                -distance.cdist(np.expand_dims(mean, axis=0), embeddings[start:end], 'minkowski', p=1.).reshape(-1)
            )
        else:
            raise ValueError("Invalid metric!")

        all_similarities[start:end] = similarities

        count += n_sample_per_class[i]

    all_similarities.sort()
    percentile_values = []
    for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        value = all_similarities[int(percentile / 100 * len(embeddings))]
        percentile_values.append(value)
    print("Percentile values:", percentile_values)

    return percentile_values
