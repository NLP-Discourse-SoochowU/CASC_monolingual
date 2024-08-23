# coding: UTF-8
import numpy as np
import random
import torch
import gensim
import torch.nn as nn
from util.file_util import *
from model.static_parameters import *
from sklearn.cluster import KMeans

import sys
sys.path.append("..")
from app_cfg import use_cuda

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


class MCluster(nn.Module):
    def __init__(self, args, label2id, comment_num, km_val):
        self.args = args
        feature_size = len(label2id.keys())
        feature_emb_ = nn.Parameter(torch.empty(feature_size, args.cfeature_space, dtype=torch.float))
        self.feature_emb = feature_emb_.to(device)
        nn.init.xavier_normal_(self.feature_emb)

        super(MCluster, self).__init__()
        self.wv_model = load_data("data/cache/w2v_embedding")
        self.session = Session(args.use_gpu, km_value=min(km_val, comment_num))
        # int((comment_num / 2.) ** 0.5) A bug of using KM in this work
        self.word_size = args.w2v_size


    def forward(self, comment_texts, feat_ids, keywords_all, comment_text_ids, model_xl=None):
        corpus_embeddings = model_xl.encode(comment_texts, batch_size=64, show_progress_bar=False,
                                            convert_to_tensor=True)
        comment_reps = corpus_embeddings.to(device)
        # feature engineering
        if self.args.comment_type == "cmt_tree":
            article_vec, tree_vec, level_vec, cmt_vec, kw_vec = None, None, None, None, None
            for c_id, comment_feat in enumerate(feat_ids):
                article_id, tree_id, tree_level, comment_id = comment_feat
                article_feat = self.feature_emb[article_id].unsqueeze(0).detach()
                article_vec = article_feat if article_vec is None else torch.cat((article_vec, article_feat), 0)
                tree_feat = self.feature_emb[tree_id].unsqueeze(0).detach()
                tree_vec = tree_feat if tree_vec is None else torch.cat((tree_vec, tree_feat), 0)
                tree_level_feat = self.feature_emb[tree_level].unsqueeze(0).detach()
                level_vec = tree_level_feat if level_vec is None else torch.cat((level_vec, tree_level_feat), 0)
                comment_id_feat = self.feature_emb[comment_id].unsqueeze(0).detach()
                cmt_vec = comment_id_feat if cmt_vec is None else torch.cat((cmt_vec, comment_id_feat), 0)

                keywords = keywords_all[c_id]
                pad_kw_feat = nn.Parameter(torch.empty(1, self.word_size, dtype=torch.float))
                nn.init.xavier_normal_(pad_kw_feat)
                kw_feat = None
                kw_feat_num = 0
                for keyword in keywords:
                    if keyword in self.wv_model.keys():
                        one_feat = torch.Tensor(self.wv_model[keyword].copy()).unsqueeze(0)
                        kw_feat = one_feat if kw_feat is None else torch.cat((kw_feat, one_feat), 0)
                        kw_feat_num += 1
                if kw_feat_num == 0:
                    kw_feat = pad_kw_feat
                else:
                    kw_feat = torch.mean(kw_feat, 0).unsqueeze(0)
                kw_vec = kw_feat if kw_vec is None else torch.cat((kw_vec, kw_feat), 0)

            if self.args.cfeature_label == 1:
                comment_reps = torch.cat((comment_reps, article_vec), -1)
            elif self.args.cfeature_label == 2:
                comment_reps = torch.cat((comment_reps, article_vec, tree_vec), -1)
            else:
                comment_reps = torch.cat((comment_reps, article_vec, tree_vec, level_vec), -1)

            if self.args.cmt_id_feature:
                comment_reps = torch.cat((comment_reps, cmt_vec), -1)  # indicate two sentences belong to one cmt
            if self.args.ckw_feature:
                comment_reps = torch.cat((comment_reps, kw_vec.cuda()), -1)

        self.session.forward(comment_texts, keywords_all, comment_text_ids, comment_reps.detach())

class Session:
    def __init__(self, use_gpu, km_value):
        self.data_points = []
        self.keywords = []
        self.data_text_ids = []
        self.data_texts = []
        self.pred_clusters = None
        self.km = KMeans(n_clusters=km_value, n_init="auto")
        self.use_gpu = use_gpu
        
    def update(self):
        clustering = self.km.fit(self.data_points)
        self.pred_clusters = clustering.labels_

    def forward(self, comment_texts, keywords_all, comment_text_ids, comment_reps):
        if self.use_gpu:
            self.data_points = self.data_points + [point.cpu().numpy() for point in comment_reps]
        else:
            self.data_points = self.data_points + [point.numpy() for point in comment_reps]
        self.keywords = self.keywords + keywords_all[:]
        self.data_text_ids = self.data_text_ids + comment_text_ids[:]
        self.data_texts = self.data_texts + comment_texts[:]
