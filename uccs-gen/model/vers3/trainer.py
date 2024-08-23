# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date: 2019.1.24
@Description: The trainer of our parser.
"""
import math
import logging
import random
import numpy as np
import progressbar
from util.DL_tricks import *
from model.static_parameters import *
from model.vers3.model import MCluster
from transformers import AutoTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score

import sys
sys.path.append("..")
from app_cfg import use_cuda

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, data_set, args, label2id, comment_num, km_val):
        self.args = args
        self.train_set = data_set
        self.model = MCluster(args, label2id, comment_num, km_val)
        self.model.eval()
        self.tokenizer = None
        self.model_xl = SentenceTransformer('all-MiniLM-L6-v2')
        self.model_xl.eval()
        if args.use_gpu:
            self.model.to(device)
            self.model_xl.to(device)
        self.comment_num = comment_num

    def train(self, logger=None):
        n_iter = 0
        batch_numbers = math.ceil(len(self.train_set) / float(BATCH_SIZE))
        p = progressbar.ProgressBar()
        p.start(batch_numbers)
        if self.args.comment_type == "cmt_list":
            batch_iter = self.gen_batch_iter()
        else:
            batch_iter = self.gen_batch_iter_trees()
        for n_batch, item in enumerate(batch_iter, start=1):
            comment_texts, keywords_all, feat_ids, comment_text_ids = item
            n_iter += 1
            p.update((n_iter % batch_numbers))
            self.model(comment_texts, feat_ids, keywords_all, comment_text_ids, self.model_xl)
        self.model.session.update() 
        p.finish()
        clusters2save, comment_labels = self.texts_to_vectors()
        return clusters2save, comment_labels

    def gen_batch_iter(self):
        random_instances = self.train_set[:]
        num_instances = len(random_instances)
        offset = 0
        while offset < num_instances:
            batch = random_instances[offset: min(num_instances, offset + BATCH_SIZE)]
            comment_texts = list()
            comment_text_cmt_ids = list()
            keywords_all = list()
            comment_feature_ids = list()

            for batch_idx, item in enumerate(batch):
                comment_id, keywords, text_item, text_cmt_id = item
                comment_texts.append(text_item)
                comment_text_cmt_ids.append(text_cmt_id)
                keywords_all.append(keywords)
                comment_feature_ids.append(comment_id)

            yield comment_texts, keywords_all, comment_feature_ids, comment_text_cmt_ids
            offset = offset + BATCH_SIZE

    def gen_batch_iter_trees(self):
        random_instances = self.train_set[:]
        num_instances = len(random_instances)
        offset = 0
        while offset < num_instances:
            batch = random_instances[offset: min(num_instances, offset + BATCH_SIZE)]
            comment_texts = list()
            comment_text_cmt_ids = list()
            keywords_all = list()
            comment_feature_ids = list()
            for batch_idx, item in enumerate(batch):
                article_id, tree_id, tree_level, comment_id, keywords, text_item, text_cmt_id = item
                comment_texts.append(text_item)
                comment_text_cmt_ids.append(text_cmt_id)
                keywords_all.append(keywords)
                comment_feature_ids.append((article_id, tree_id, tree_level, comment_id))
            yield comment_texts, keywords_all, comment_feature_ids, comment_text_cmt_ids
            offset = offset + BATCH_SIZE

    def evaluate(self):
        """ ACC, NMI and MCC
            Evaluate according to the predicted and gold clusters in the model
        """
        nmi_score = normalized_mutual_info_score(self.model.session.pred_clusters, self.model.session.gold_clusters)
        print_("score: " + str(nmi_score), clog_path)

    def texts_to_vectors(self):
        """ transform texts to representations
        """
        session = self.model.session  # trained session
        clusters_all = []
        group_num = max(session.pred_clusters)
        sent_groups = [0 for _ in range(group_num + 1)]
        for id_ in session.pred_clusters:
            sent_groups[id_] += 1

        id2new = dict()
        new_idx = 0
        for idx, item in enumerate(sent_groups):
            if item > 1 and idx not in id2new.keys():
                id2new[idx] = new_idx
                new_idx += 1
        # build clusters
        comment_labels = []
        sent_groups = [[] for _ in range(new_idx + 1)]
        kw_groups = [[] for _ in range(new_idx + 1)]
        sent_cmt_id_groups = [[] for _ in range(new_idx + 1)]
        other_count = 0
        for kws, sent, id_, sent_cmt_id in zip(session.keywords, session.data_texts, session.pred_clusters, session.data_text_ids):
            if id_ in id2new.keys():
                new_i = id2new[id_]
                comment_labels.append(new_i)
                sent_groups[new_i].append(sent)
                kw_groups[new_i].append(kws)
                sent_cmt_id_groups[new_i].append(sent_cmt_id)
            else:
                other_count += 1
                # Others
                comment_labels.append(new_idx)
                sent_groups[new_idx].append(sent)
                kw_groups[new_idx].append(kws)
                sent_cmt_id_groups[new_idx].append(sent_cmt_id)
        print("OTHERS: ", other_count)
        for sent_g, kw_g, sent_cmt_ids_g in zip(sent_groups, kw_groups, sent_cmt_id_groups):
            clusters_all.append((sent_g, kw_g, sent_cmt_ids_g))
        return clusters_all, comment_labels
