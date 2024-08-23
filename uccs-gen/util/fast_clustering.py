import torch
from numpy import mean
from torch import Tensor, device
import math
import os
from util.file_util import save_data, load_data


def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def get_curve(y1, y2, estimate_number):
    """ The higher, the teeper
    """
    y1_c_number = 50.
    k1 = (y2 ** 2 - y1 ** 2) / (estimate_number - y1_c_number)
    k2 = 0 if k1 == 0 else (y2 ** 2 - estimate_number * k1) / k1
    return k1, k2


def community_detection(embeddings, threshold=0.55, min_community_size=1, batch_size=1024, max_delta_thr=0.9, estimate_thr=0.85, estimate_num=200, max_cluster_size=500, bg_stems=None, sentences_stems=None):
    """ Function for Fast Community Detection
        Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
        Returns only communities that are larger than min_community_size. The communities are returned
        in decreasing order. The first element in each list is the central point in the community.
    """
    k1, k2 = get_curve(threshold, estimate_thr, estimate_num)
    # print(k1, k2)

    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    # print(embeddings.size())
    threshold = torch.tensor(threshold, device=embeddings.device)

    extracted_communities = []
    extracted_community_values = dict()
    dict_idx = 0

    # Maximum size for community
    increase_speed = 10  # Final 10
    min_community_size = min(min_community_size, len(embeddings))
    sort_max_size = min(max(2 * min_community_size, 50), len(embeddings))

    # num2thr_ana = list()
    for start_idx in range(0, len(embeddings), batch_size):
        # Compute cosine similarity scores
        cos_scores = cos_sim(embeddings[start_idx:start_idx + batch_size], embeddings)
        center_to_bg_scores = None if sentences_stems is None or bg_stems is None else [len(item & bg_stems) for item in sentences_stems[start_idx:start_idx + batch_size]]

        # Minimum size for a community
        top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

        # Filter for rows >= min_threshold
        for i in range(len(top_k_values)):
            c2b_score = 1 if center_to_bg_scores is None else max(math.log2(center_to_bg_scores[i] + 1), 0.1)
            sort_max_size_ = sort_max_size  # The original algorithm does not do this.
            if top_k_values[i][-1] >= threshold:
                new_cluster = []
                new_cluster_values = []

                threshold_dy = threshold.clone().item()

                # Only check top k most similar entries
                top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size_, largest=True)

                # Check if we need to increase sort_max_size_, we do not think it's good for a cc with more than 300 
                last_val = None
                while top_val_large[-1] > threshold_dy and sort_max_size_ < len(embeddings) and sort_max_size_ < max_cluster_size and sort_max_size_ != last_val:
                    last_val = sort_max_size_  # to avoid none-end loop
                    sort_max_size_ = min(sort_max_size_ + increase_speed, len(embeddings))
                    top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size_, largest=True)
                    threshold_dy = threshold_dy if k1 == k2 == 0 else (k1 * (sort_max_size_ + k2)) ** 0.5
                    threshold_dy = min(threshold_dy, max_delta_thr)
                for idx, val in zip(top_idx_large.tolist(), top_val_large):
                    if val < threshold_dy:
                        break
                    new_cluster.append(idx)
                    new_cluster_values.append(val)
                    if len(new_cluster) >= max_cluster_size:
                        break
                # num2thr_ana.append((threshold_dy, c2b_score, torch.mean(torch.Tensor(new_cluster_values)), len(new_cluster)))
                extracted_communities.append(new_cluster)
                extracted_community_values[dict_idx] = c2b_score * torch.mean(torch.Tensor(new_cluster_values)) * len(new_cluster) 
                dict_idx += 1
        del cos_scores
    extracted_community_values = sorted(extracted_community_values.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)

    # Higher quality clusters first
    extracted_communities = [extracted_communities[item[0]] for item in extracted_community_values]
    # extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for cluster_id, community in enumerate(extracted_communities):
        # community = sorted(community)  # No need to sord inside the comment cluster
        non_overlapped_community = []
        for idx in community:
            if idx not in extracted_ids:
                non_overlapped_community.append(idx)

        if len(non_overlapped_community) >= min_community_size:
            unique_communities.append(non_overlapped_community)
            extracted_ids.update(non_overlapped_community)

    unique_communities = sorted(unique_communities, key=lambda x: len(x), reverse=True)

    # save_data(num2thr_ana, "data/mysql_db/cc_size_thr_ana.pkl")
    # input("Done")
    # print([len(item) for item in unique_communities[:10]])
    return unique_communities


def community_detection_init(embeddings, threshold=0.75, min_community_size=1, batch_size=1024, max_delta_thr=0.9, estimate_thr=0.85, estimate_num=200, max_cluster_size=500, bg_stems=None, sentences_stems=None):
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)

    threshold = torch.tensor(threshold, device=embeddings.device)

    extracted_communities = []

    # Maximum size for community
    min_community_size = min(min_community_size, len(embeddings))
    sort_max_size = min(max(2 * min_community_size, 50), len(embeddings))

    for start_idx in range(0, len(embeddings), batch_size):
        # Compute cosine similarity scores
        cos_scores = cos_sim(embeddings[start_idx:start_idx + batch_size], embeddings)

        # Minimum size for a community
        top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

        # Filter for rows >= min_threshold
        for i in range(len(top_k_values)):
            if top_k_values[i][-1] >= threshold:
                new_cluster = []

                # Only check top k most similar entries
                top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                # Check if we need to increase sort_max_size
                while top_val_large[-1] > threshold and sort_max_size < len(embeddings):
                    sort_max_size = min(2 * sort_max_size, len(embeddings))
                    top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                for idx, val in zip(top_idx_large.tolist(), top_val_large):
                    if val < threshold:
                        break

                    new_cluster.append(idx)

                extracted_communities.append(new_cluster)

        del cos_scores
    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for cluster_id, community in enumerate(extracted_communities):
        community = sorted(community)
        non_overlapped_community = []
        for idx in community:
            if idx not in extracted_ids:
                non_overlapped_community.append(idx)

        if len(non_overlapped_community) >= min_community_size:
            unique_communities.append(non_overlapped_community)
            extracted_ids.update(non_overlapped_community)

    unique_communities = sorted(unique_communities, key=lambda x: len(x), reverse=True)
    return unique_communities