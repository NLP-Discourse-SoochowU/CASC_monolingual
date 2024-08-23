# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR
import torch
import torch.nn as nn
import sys
import progressbar
import gensim
from util.file_util import load_data, get_stem
from util.fast_clustering import community_detection
# from util.fast_clustering import community_detection_init as community_detection

sys.path.append("..")
from app_cfg import use_cuda

if not use_cuda:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


def upd_clusters(ori_clusters):
    new_clusters = list()
    other_comments = list()
    for cluster in ori_clusters:
        if len(cluster) == 1:
            other_comments += cluster
        else:
            new_clusters.append(cluster)
    return new_clusters, other_comments


class Trainer:
    def __init__(self, data_set, args, label2id, comment_num, background_info, sent_rep_name):
        self.sent_rep_name = sent_rep_name
        if sent_rep_name == "all-MiniLM-L6-v2":
            self.model = SentenceTransformer(sent_rep_name).to(device)
        else:
            self.model = INSTRUCTOR(sent_rep_name).to(device)
        self.train_set = data_set
        self.args = args
        feature_size = len(label2id.keys())
        feature_emb_ = nn.Parameter(torch.empty(feature_size, args.cfeature_space, dtype=torch.float))
        self.feature_emb = feature_emb_.to(device)
        self.comment_num = comment_num
        nn.init.xavier_normal_(self.feature_emb)
        self.wv_model = load_data("data/cache/w2v_embedding")
        self.word_size = args.w2v_size
        self.bg_info = get_stem(background_info)

    def train_cmt(self, logger):
        article_vec, tree_vec, level_vec, cmt_vec, kw_vec = None, None, None, None, None
        p = progressbar.ProgressBar()
        p.start(len(self.train_set))
        p_value = 0
        keywords_all = list()
        corpus_sentences = list()
        sentences_cmt_ids = list()
        for item in self.train_set:
            p_value += 1
            p.update(p_value)
            article_id, tree_id, tree_level, comment_id, keywords, text_item, text_cmt_id = item
            keywords_all.append(keywords)
            corpus_sentences.append(text_item)
            sentences_cmt_ids.append(text_cmt_id)

            article_feat = self.feature_emb[article_id].unsqueeze(0).detach()
            article_vec = article_feat if article_vec is None else torch.cat((article_vec, article_feat), 0)
            tree_feat = self.feature_emb[tree_id].unsqueeze(0).detach()
            tree_vec = tree_feat if tree_vec is None else torch.cat((tree_vec, tree_feat), 0)
            tree_level_feat = self.feature_emb[tree_level].unsqueeze(0).detach()
            level_vec = tree_level_feat if level_vec is None else torch.cat((level_vec, tree_level_feat), 0)
            comment_feat = self.feature_emb[comment_id].unsqueeze(0).detach()
            cmt_vec = comment_feat if cmt_vec is None else torch.cat((cmt_vec, comment_feat), 0)

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
        p.finish()
        logger.info("Encoding & clustering comments in an article cluster...")
        instruction = "Represent the user comment for clustering;"
        if self.sent_rep_name != "all-MiniLM-L6-v2":
            corpus_sentences_ = [[instruction, sent_item] for sent_item in corpus_sentences]
        else:
            corpus_sentences_ = corpus_sentences[:]
        corpus_embeddings = self.model.encode(corpus_sentences_, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
        corpus_embeddings = corpus_embeddings.to(device)
        corpus_word_stems = [get_stem(sentence) for sentence in corpus_sentences]
        
        if self.args.cfeature_label == 1:
            corpus_embeddings = torch.cat((corpus_embeddings, article_vec), -1)
        elif self.args.cfeature_label == 2:
            corpus_embeddings = torch.cat((corpus_embeddings, article_vec, tree_vec), -1)
        else:
            corpus_embeddings = torch.cat((corpus_embeddings, article_vec, tree_vec, level_vec), -1)

        if self.args.cmt_id_feature:
            corpus_embeddings = torch.cat((corpus_embeddings, cmt_vec), -1)  # indicate two sentences belong to one cmt
        if self.args.ckw_feature:
            corpus_embeddings = torch.cat((corpus_embeddings, kw_vec.to(device)), -1)

        comment_num_c = corpus_embeddings.size()[0]
        if comment_num_c == 1:
            clusters = [[0]]
        else:
            thr_min = self.args.similar_max_min[0]
            thr_delta = self.args.similar_max_min[1]
            thr_est = self.args.similar_max_min[2]
            thr_est_num = self.args.similar_max_min[3]
            clusters = community_detection(corpus_embeddings, threshold=thr_min, max_delta_thr=thr_delta, estimate_thr=thr_est, estimate_num=thr_est_num, bg_stems=self.bg_info, sentences_stems=corpus_word_stems)
            
        clusters, other_comment_ids = upd_clusters(clusters)

        cluster_id = 0
        clusters2save = []
        comment_labels = [-1 for _ in range(self.comment_num)]
        for cluster in clusters:
            sent_ids = cluster[:]
            selected_ids = list()
            kw_list, sentences, sentences_ids = list(), list(), list()
            for sent_id in sent_ids:
                if len(keywords_all[sent_id]) == 0:
                    other_comment_ids.append(sent_id)
                else:
                    comment_labels[sent_id] = cluster_id
                    kw_list.append(keywords_all[sent_id])
                    sentences.append(corpus_sentences[sent_id])
                    sentences_ids.append(sentences_cmt_ids[sent_id])
                    selected_ids.append(sent_id)

            if len(selected_ids) > 1:
                clusters2save.append((sentences, kw_list, sentences_ids))
                cluster_id += 1
            else:
                other_comment_ids += selected_ids

        # build the other comment cluster
        kw_list, sentences, sentences_ids = list(), list(), list()

        for idx_flag, sent_id in enumerate(other_comment_ids):
            comment_labels[sent_id] = cluster_id
            kw_list.append(keywords_all[sent_id])
            if idx_flag == 0:
                sentences.append("$$Other comments.$$ " + corpus_sentences[sent_id])
            else:
                sentences.append(corpus_sentences[sent_id])
            sentences_ids.append(sentences_cmt_ids[sent_id])
        clusters2save.append((sentences, kw_list, sentences_ids))
        assert -1 not in comment_labels
        return clusters2save, comment_labels

    def train_plain(self, logger):
        cmt_vec, kw_vec = None, None
        keywords_all = list()
        corpus_sentences = list()
        sentences_cmt_ids = list()

        for item in self.train_set:
            comment_id, keywords, text_item, text_cmt_id = item
            keywords_all.append(keywords)
            corpus_sentences.append(text_item)
            sentences_cmt_ids.append(text_cmt_id)

            # comment id feature
            comment_feat = self.feature_emb[comment_id].unsqueeze(0).detach()
            cmt_vec = comment_feat if cmt_vec is None else torch.cat((cmt_vec, comment_feat), 0)
            # keyword feature
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

        logger.info("Encoding & clustering comments in an article cluster...")
        instruction = "Represent the user comment for clustering;"
        if self.sent_rep_name != "all-MiniLM-L6-v2":
            corpus_sentences_ = [[instruction, sent_item] for sent_item in corpus_sentences]
        else:
            corpus_sentences_ = corpus_sentences[:]
        corpus_embeddings = self.model.encode(corpus_sentences_, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
        corpus_embeddings = corpus_embeddings.to(device)
        corpus_word_stems = [get_stem(sentence) for sentence in corpus_sentences]

        if self.args.cmt_id_feature:
            corpus_embeddings = torch.cat((corpus_embeddings, cmt_vec), -1)  # indicate two sentences belong to one cmt
        if self.args.ckw_feature:
            corpus_embeddings = torch.cat((corpus_embeddings, kw_vec.to(device)), -1)

        comment_num_c = corpus_embeddings.size()[0]
        if comment_num_c == 1:
            clusters = [[0]]
        else:
            thr_min = self.args.similar_max_min[0]
            thr_delta = self.args.similar_max_min[1]
            thr_est = self.args.similar_max_min[2]
            thr_est_num = self.args.similar_max_min[3]
            clusters = community_detection(corpus_embeddings, threshold=thr_min, max_delta_thr=thr_delta, estimate_thr=thr_est, estimate_num=thr_est_num, bg_stems=self.bg_info, sentences_stems=corpus_word_stems)

        clusters, other_comment_ids = upd_clusters(clusters)

        cluster_id = 0
        clusters2save = list()
        comment_labels = [-1 for _ in range(self.comment_num)]
        for cluster in clusters:
            comment_ids = cluster[:]
            selected_ids = list()
            kw_list, sentences, sentences_ids = list(), list(), list()
            for sent_id in comment_ids:
                if len(keywords_all[sent_id]) == 0:
                    other_comment_ids.append(sent_id)
                else:
                    comment_labels[sent_id] = cluster_id
                    kw_list.append(keywords_all[sent_id])
                    sentences.append(corpus_sentences[sent_id])
                    sentences_ids.append(sentences_cmt_ids[sent_id])
                    selected_ids.append(sent_id)

            if len(selected_ids) > 1:
                clusters2save.append((sentences, kw_list, sentences_ids))
                cluster_id += 1
            else:
                other_comment_ids += selected_ids

        # build the other comment cluster
        kw_list, sentences, sentences_ids = list(), list(), list()

        for idx_flag, sent_id in enumerate(other_comment_ids):
            comment_labels[sent_id] = cluster_id
            kw_list.append(keywords_all[sent_id])
            if idx_flag == 0:
                sentences.append("$$Other comments.$$ " + corpus_sentences[sent_id])
            else:
                sentences.append(corpus_sentences[sent_id])
            sentences_ids.append(sentences_cmt_ids[sent_id])

        clusters2save.append((sentences, kw_list, sentences_ids))
        assert -1 not in comment_labels
        return clusters2save, comment_labels

    def train(self, logger):
        if self.args.comment_type == "cmt_list":
            clusters2save, comment_labels = self.train_plain(logger)
        elif self.args.comment_type == "cmt_tree":
            clusters2save, comment_labels = self.train_cmt(logger)
        else:
            clusters2save = comment_labels = None
            try:
                sys.exit(0)
            except:
                logger.error("The system only know two kinds of comment structure: cmt_list and cmt_tree.")
        return clusters2save, comment_labels
