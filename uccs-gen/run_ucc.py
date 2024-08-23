import argparse
from string import punctuation
from model.static_parameters import *
from util.file_util import *
import numpy as np
import torch
import sys
import random
from model.vers1.trainer import Trainer as Trainer1  
from model.vers2.trainer import Trainer as Trainer2  
from model.vers3.trainer import Trainer as Trainer3  
from model.vers4.trainer import Trainer as Trainer4 
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import gensim
from sentence_transformers import SentenceTransformer, util
import nltk
from itertools import combinations
from nltk.corpus import stopwords
import pymysql
import pandas as pd
import csv
import json
import logging
from db_conn import dsta_db
sys.path.append("..")
from app_cfg import *
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from llm_model import summary_gen, title_sum_gen

# SEED = 3  # 3 27
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

logging.basicConfig(level=logging.INFO, filename='logs/clustering.log', filemode='a', 
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
nlp = spacy.load('en_core_web_sm')
db_obj = dsta_db(host, username, passwd, port, db_name) if use_db else None
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")


class MySentences(object):
    def __init__(self, sentences):
        self.sentences = sentences

    def __iter__(self):
        for line in self.sentences:
            yield line.split()

def form_set(cmt_sent_lines, cmt_ids, feat_all, label2id, keywords_all, data_type):
    data_set = []
    if data_type == "cmt_list":
        for sent_id, sentence in enumerate(cmt_sent_lines):
            comment_id = feat_all[sent_id]
            sentence_cmt_id = cmt_ids[sent_id]
            keywords = keywords_all[sent_id]
            data_set.append((label2id[comment_id], keywords, sentence, sentence_cmt_id))
    elif data_type == "cmt_tree":
        for sent_id, sentence in enumerate(cmt_sent_lines):
            parts = feat_all[sent_id].split("$^$")
            article_id = parts[0]  # article id
            tree_id = article_id + parts[1]  # tree id
            tree_level = tree_id + parts[2]  # tree_level
            comment_id = tree_level + parts[3]
            sentence_cmt_id = cmt_ids[sent_id]
            keywords = keywords_all[sent_id]
            data_set.append((label2id[article_id], label2id[tree_id], label2id[tree_level], label2id[comment_id],
                             keywords, sentence, sentence_cmt_id))
    else:
        logger.error("Please check the value of the attribution 'comment_type'!")
        sys.exit(0)
    return data_set


def get_background_info(args, article_ids, ac_id_count=None):
    if not args.use_db:
        with open(name2background, "r") as f:
            name2background_dict = json.load(f)
        if args.llm_bg:
            rand_size = min(len(article_ids), 50)
            rand_names = random.sample(article_ids, rand_size)
            titles = [name2background_dict[name] for name in rand_names]
            background_info = title_sum_gen(titles, args.llm_name)
        else:
            rand_size = min(len(article_ids), 6)
            rand_names = random.sample(article_ids, rand_size)
            background_info = " ".join([name2background_dict[name] for name in rand_names])

            # # DEV Test title as BG
            # bg_1 = [
            #     "Trump's Russia Motives",
            #     "Comey's Haunting News on Trump",
            #     "Connecting Trump's Dots to Russia",
            #     "Unraveling Trump's Motivations Toward Russia",
            #     "Trump's Revelations: The Impact of Comey's Disturbing News",
            #     "Tracing the Links: Connecting Trump and Russia",
            #     "Motivations Behind Trump's Relationship with Russia",
            #     "Comey's Disturbing Revelation Regarding Trump",
            #     "Tracing the Links Between Trump and Russia"
            # ]
            # bg_2 = [
            #     "6 polyclinics open for 2 weekends for patients with acute Covid-19 symptoms",
            #     "Two-Weekend Opening of Six Polyclinics for Acute Covid-19 Cases",
            #     "Acute Covid-19 Symptoms: Six Polyclinics Operating Across Two Weekends",
            #     "Specialized Care for Acute Covid-19 Cases: Six Polyclinics Open Over Two Weekends",
            #     "Weekend Initiative: Six Polyclinics Available for Patients with Acute Covid-19 Symptoms",
            #     "Extended Service: Six Polyclinics Open for Two Weekends to Address Acute Covid-19 Cases",
            #     "Managing Acute Covid-19: Six Polyclinics Extend Services Over Two Weekends",
            #     "Two-Weekend Access: Six Polyclinics for Patients with Acute Covid-19 Symptoms",
            #     "Responding to Acute Cases: Six Polyclinics Operating During Two Weekends for Covid-19"
            # ]
            # bg_3 = [
            #     "27,000 children have had Covid-19 since Omicron wave started: MOH",
            #     "MOH Reports Over 27,000 Children Contracted Covid-19 During Omicron Surge",
            #     "Omicron Wave Impact: More than 27,000 Children Affected by Covid-19, MOH States",
            #     "MOH: 27,000 Children Infected with Covid-19 Amidst Omicron Outbreak",
            #     "Covid-19 Surge in Children: MOH Confirms 27,000 Cases During Omicron Wave",
            #     "Impact of Omicron Wave: MOH Reports Covid-19 in 27,000 Children",
            #     "Omicron Wave's Toll: MOH Acknowledges 27,000 Children with Covid-19",
            #     "MOH Announcement: 27,000 Children Contract Covid-19 During Omicron Onset",
            #     "Children and Covid-19: MOH Reports Over 27,000 Cases Since Omicron Wave Commenced"
            # ]
            # bg_num = args.km_val
            # bg_all = [bg_1, bg_2, bg_3]
            # rand_bg = random.sample(bg_all[ac_id_count], bg_num)
            # background_info = " ".join(rand_bg)
    else:
        background_info = list()
        article_info = db_obj.get_article_by_ids(article_ids)
        if args.llm_bg:
            rand_size = min(len(article_info), 50)
            rand_info = random.sample(article_info, rand_size)
            titles = [item[0] for item in rand_info]
            background_info = title_sum_gen(titles, args.llm_name)
        else:
            rand_size = min(len(article_info), 6)
            rand_info = random.sample(article_info, rand_size)
            background_info = " ".join([item[0] for item in rand_info])
    return background_info


def build_tree_comments(article_cluster, data_type, word_size):
    label2id = dict()
    label_id = 0
    comments_all, feat_all = list(), list()
    unique_comment_ids = list()
    if data_type == "cmt_list":
        with open(carticle_name2comments) as f_:
            reader_ = csv.DictReader(f_)
            for item_ in reader_:
                for article_id in article_cluster:
                    if item_["cmt_article_id"] == article_id:
                        comment_feat_one = str(item_["cmt_id"])
                        comment_txt = item_["cmt_content"].replace("\n", "").strip().lower()
                        comments_all.append(comment_txt)
                        unique_comment_ids.append(item_["cmt_id"])
                        feat_all.append(comment_feat_one)

                        if comment_feat_one not in label2id.keys():
                            label2id[comment_feat_one] = label_id
                            label_id += 1
    elif data_type == "cmt_tree":
        with open(carticle_name2trees) as f_:
            reader_ = csv.DictReader(f_)
            for item_ in reader_:
                for article_id in article_cluster:
                    if item_["cmt_article_id"] == article_id:
                        comment_feat_one = item_["cmt_tree"]
                        comment_txt = item_["cmt_content"].replace("\n", "").strip().lower()
                        comments_all.append(comment_txt)
                        unique_comment_ids.append(item_["cmt_id"])
                        feat_all.append(comment_feat_one)

                        parts = comment_feat_one.split("$^$")
                        article_id = parts[0]
                        tree_id = article_id + parts[1]
                        tree_level = tree_id + parts[2]
                        comment_id = tree_level + parts[3]
                        if article_id not in label2id.keys():
                            label2id[article_id] = label_id
                            label_id += 1
                        if tree_id not in label2id.keys():
                            label2id[tree_id] = label_id
                            label_id += 1
                        if tree_level not in label2id.keys():
                            label2id[tree_level] = label_id
                            label_id += 1
                        if comment_id not in label2id.keys():
                            label2id[comment_id] = label_id
                            label_id += 1
    else:
        logger.error("Please check the value of the attribution 'comment_type'!")
        sys.exit(0)

    if len(comments_all) == 0:
        sentences = MySentences(["PAD."])
        wv_model = gensim.models.Word2Vec(sentences, size=word_size, window=5, min_count=1, workers=4)
    else:
        sentences = MySentences(comments_all)
        wv_model = gensim.models.Word2Vec(sentences, size=word_size, window=5, min_count=1, workers=4)
    return comments_all, unique_comment_ids, feat_all, label2id, wv_model


def build_tree_comments_db(article_cluster, data_type, word_size):
    label2id = dict()
    label_id = 0
    comments_all, feat_all = list(), list()
    unique_comment_ids = list()
    if data_type == "cmt_list":
        comments_db = db_obj.get_comments_by_article_ids(article_cluster)
        print(f"Done selecting comments by article_ids, with {len(comments_db)} comments.")
        if comments_db is not None:
            for comment_db in comments_db:
                unique_comment_ids.append(comment_db[0])
                comment_txt = comment_db[1].replace("\n", "").strip().lower()
                comments_all.append(comment_txt)
                comment_feat_one = str(comment_db[0])
                feat_all.append(comment_feat_one)
                if comment_feat_one not in label2id.keys():
                    label2id[comment_feat_one] = label_id
                    label_id += 1
    elif data_type == "cmt_tree":
        comments_db = db_obj.get_comment_trees_by_article_ids(article_cluster)
        print(f"Done selecting comments by article_ids, with {len(comments_db)} comments.")
        if comments_db is not None:
            for comment_db in comments_db:
                unique_comment_ids.append(comment_db[0])
                comment_txt = comment_db[1].replace("\n", "").strip().lower()
                comments_all.append(comment_txt)
                comment_feat_one = comment_db[2]
                feat_all.append(comment_feat_one)
                parts = comment_feat_one.split("$^$")
                article_id = parts[0]
                tree_id = article_id + parts[1]
                tree_level = tree_id + parts[2]
                comment_id = tree_level + parts[3]
                if article_id not in label2id.keys():
                    label2id[article_id] = label_id
                    label_id += 1
                if tree_id not in label2id.keys():
                    label2id[tree_id] = label_id
                    label_id += 1
                if tree_level not in label2id.keys():
                    label2id[tree_level] = label_id
                    label_id += 1
                if comment_id not in label2id.keys():
                    label2id[comment_id] = label_id
                    label_id += 1
    else:
        logger.error("Please check the value of the attribution 'comment_type'!")
        sys.exit(0)

    if len(comments_all) == 0:
        sentences = MySentences(["PAD."])
        wv_model = gensim.models.Word2Vec(sentences, size=word_size, window=5, min_count=1, workers=4)
    else:
        sentences = MySentences(comments_all)
        wv_model = gensim.models.Word2Vec(sentences, size=word_size, window=5, min_count=1, workers=4)
    return comments_all, unique_comment_ids, feat_all, label2id, wv_model


def build_eval_comments(ann_comments_all, ann_comment_ids_all, ann_feats_all, comment_type="cmt_list", word_size=16):
    label2id, label_id = dict(), 0
    comments_all, comment_ids_all, feat_all = list(), list(), list()
    for cc, c_ids, cc_feat in zip(ann_comments_all, ann_comment_ids_all, ann_feats_all):
        for comment, comment_id, feat in zip(cc, c_ids, cc_feat):
            comment_txt = comment.strip().lower()
            comment_txt = comment_txt.replace("\n", "")
            comments_all.append(comment_txt)
            comment_ids_all.append(comment_id)
            # print(feat)
            parts = feat.split("$^$")
            article_id = parts[0]
            tree_id = article_id + parts[1]
            tree_level = tree_id + parts[2]
            comment_id = tree_level + parts[3]

            if comment_type == "cmt_tree":
                feat_all.append(feat)
                if article_id not in label2id.keys():
                    label2id[article_id] = label_id
                    label_id += 1
                if tree_id not in label2id.keys():
                    label2id[tree_id] = label_id
                    label_id += 1
                if tree_level not in label2id.keys():
                    label2id[tree_level] = label_id
                    label_id += 1
            else:
                feat_all.append(comment_id)
        
            if comment_id not in label2id.keys():
                label2id[comment_id] = label_id
                label_id += 1

    if len(comments_all) == 0:
        sentences = MySentences(["PAD."])
        wv_model = gensim.models.Word2Vec(sentences, size=word_size, window=5, min_count=1, workers=4)
    else:
        sentences = MySentences(comments_all)
        wv_model = gensim.models.Word2Vec(sentences, size=word_size, window=5, min_count=1, workers=4)
    return comments_all, comment_ids_all, feat_all, label2id, wv_model


def parse_eval_data(manual_ann_path):
    with open(manual_ann_path, "r") as f_:
        data_dict_ = json.load(f_)
        ann_acs_ = data_dict_["article_clusters"]
    ac_names_, semantic_spans, semantic_span_ids, spans_feat, ann_comments_labels, summaries = [], [], [], [], [], []
    for ac_ in ann_acs_:
        ac_names_.append(ac_["article_ids"])
        semantic_spans_one, semantic_span_ids_one, spans_feat_one, summaries_one, comment_labels_one = [], [], [], [], []
        comment_clusters = ac_["comment_clusters"]
        for cc_ in comment_clusters:
            cc_comments = list()
            cc_comment_ids = list()
            cc_comments_tree_info = list()
            for comment_with_id in cc_["comments"]:
                cc_comments.append(comment_with_id["cmt_content"])
                cc_comment_ids.append(comment_with_id["cmt_id"])
                cc_comments_tree_info.append(comment_with_id["cmt_tree"])
            semantic_spans_one.append(cc_comments)
            semantic_span_ids_one.append(cc_comment_ids)
            spans_feat_one.append(cc_comments_tree_info)
            summaries_one.append(cc_["summary"])
            comment_labels_one += [cc_["cc_id"] for _ in range(len(cc_comments))]
        semantic_spans.append(semantic_spans_one)
        semantic_span_ids.append(semantic_span_ids_one)
        spans_feat.append(spans_feat_one)
        summaries.append(summaries_one)
        ann_comments_labels.append(comment_labels_one)
    return ac_names_, semantic_spans, semantic_span_ids, spans_feat, ann_comments_labels, summaries


def clustering(data_set, args, label2id, background_info):
    comment_num = len(data_set)
    if args.cversion == 1:
        trainer = Trainer1(data_set, args, label2id, comment_num, background_info, args.sent_rep_name)
    elif args.cversion == 2:
        trainer = Trainer2(data_set, args, label2id, comment_num)
    elif args.cversion == 3:
        trainer = Trainer3(data_set, args, label2id, comment_num, args.km_val)
    else:
        trainer = Trainer4(data_set, args, label2id, comment_num, background_info, args.sent_rep_name)
    clusters, pre_ucc_labels = trainer.train(logger)
    return clusters, pre_ucc_labels


def clustering_score_prf(gold_clusters, pred_clusters):
    """ Pairwise P R F1 scores. Get micro or macro results? Micro is better with such a small number of ACs but a large number of CCs.
    """
    assert len(pred_clusters) == len(gold_clusters)
    pred_comment_clusters = dict()
    gold_comment_clusters = dict()
    c_id = 0
    for pre_cc_id, gold_cc_id in zip(pred_clusters, gold_clusters):
        if pre_cc_id in pred_comment_clusters.keys():
            pred_comment_clusters[pre_cc_id].append(c_id)
        else:
            pred_comment_clusters[pre_cc_id] = [c_id]
        if gold_cc_id in gold_comment_clusters.keys():
            gold_comment_clusters[gold_cc_id].append(c_id)
        else:
            gold_comment_clusters[gold_cc_id] = [c_id]
        c_id += 1
    g_other_id = max(gold_comment_clusters.keys())
    p_other_id = max(pred_comment_clusters.keys())
    
    # build combinations
    pred_combinations = list()
    for key_ in pred_comment_clusters.keys():
        if key_ == p_other_id:
            continue
        item_ = pred_comment_clusters[key_]
        sorted_cmb_list = [sorted(cmb_item) for cmb_item in list(combinations(item_, 2))]
        pred_combinations += sorted_cmb_list
    gold_combinations = list()
    for key_ in gold_comment_clusters.keys():
        if key_ == g_other_id:
            continue
        item_ = gold_comment_clusters[key_]
        sorted_cmb_list = [sorted(cmb_item) for cmb_item in list(combinations(item_, 2))]
        gold_combinations += sorted_cmb_list
    # P R F calculation
    its_num = float(len([item_ for item_ in pred_combinations if item_ in gold_combinations]))
    pred_num = float(len(pred_combinations))
    gold_num = float(len(gold_combinations))
    p_ = its_num / pred_num if pred_num > 0. else 0.
    r_ = its_num / gold_num if gold_num > 0. else 0.
    f_ = 2 * p_ * r_ / (p_ + r_)
    return f_


def clustering_score(pred_clusters, gold_clusters):
    # print(len(pred_clusters))
    assert len(pred_clusters) == len(gold_clusters)
    # v_score = metrics.v_measure_score(gold_clusters, pred_clusters)
    nmi_score = metrics.normalized_mutual_info_score(gold_clusters, pred_clusters)
    # f_score = clustering_score_prf(gold_clusters, pred_clusters)
    return None, nmi_score, None


def prepare_summarization(clustered_ucc):
    ucc_lines = list()
    ucc_line_info = list()
    cc_all = len(clustered_ucc)

    for cc_id, item in enumerate(clustered_ucc):
        one_cluster_comments = item[0]
        keywords = item[1]
        one_cluster_comment_ids = item[2]
        # one_cc = ["[Keywords--" + "--".join(kw) + "]  " + sent for kw, sent in zip(keywords, one_cluster_comments)]
        one_cc = one_cluster_comments[:]
        ucc_lines.append(" <S_SEP> ".join(one_cc))
        ucc_line_info.append((1, one_cluster_comment_ids))

    if cc_all == 0 or "$$Other comments.$$ " not in ucc_lines[-1]:
        ucc_lines.append("$$Other comments.$$ ")
        ucc_line_info.append((1, [-1]))
    return ucc_lines, ucc_line_info


def prepare_eval_summarization(clustered_ucc, clustered_cmt_ids, ann_sum_all=None, keywords_all=None):
    ucc_lines = list()
    ucc_line_info = list()  # count info and cmt_id
    gold_sum_texts = list()
    cc_all = len(clustered_ucc)
    for cc_id, item_ in enumerate(clustered_ucc):
        one_cluster_comments = item_
        one_cluster_comment_ids = clustered_cmt_ids[cc_id]
        comment_num = len(item_)

        keywords = keywords_all[:comment_num]
        keywords_all = keywords_all[comment_num:]
        # one_cc = ["[Keywords--" + "--".join(kw) + "]  " + sent for kw, sent in zip(keywords, one_cluster_comments)]
        one_cc = one_cluster_comments[:]
        ucc_lines.append(" <S_SEP> ".join(one_cc))
        ucc_line_info.append((1, one_cluster_comment_ids))
        cc_summary = ann_sum_all[cc_id].replace("\n", " ").strip()
        gold_sum_texts.append(cc_summary)
    ucc_lines[-1] = "$$Other comments.$$ " + ucc_lines[-1]
    return ucc_lines, gold_sum_texts, ucc_line_info


def build_eval_keywords(sum_comment_sentences):
    keywords_all = list()
    if len(sum_comment_sentences) > 0:
        # keywords
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(sum_comment_sentences)
        feature_names = vectorizer.get_feature_names_out()
        dense = vectors.todense()
        dense_list = dense.tolist()
        print_idx = 0
        for tf_idf_scores in dense_list:
            print_all = dict()
            for word, score in zip(feature_names, tf_idf_scores):
                if score > 0:
                    print_all[score] = word
            sorted_scores = sorted(list(print_all.keys()), reverse=True)
            sorted_words = [print_all[score] for score in sorted_scores]
            nn_words, vb_words = list(), list()
            keywords_nn, keywords_vb = list(), list()
            pos_out = nltk.pos_tag(nltk.word_tokenize(sum_comment_sentences[print_idx]))
            for word_pos in pos_out:
                if word_pos[0] in stopwords.words('english') or word_pos[0] in punctuation:
                    continue
                if word_pos[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
                    nn_words.append(word_pos[0])
                if word_pos[1] in ['VB', 'VBD', 'VBP', 'VBN', 'VBG', 'VBZ']:
                    vb_words.append(word_pos[0])

            for ti_words in sorted_words:
                if ti_words in nn_words:
                    keywords_nn.append(ti_words)
                if ti_words in vb_words:
                    keywords_vb.append(ti_words)
            keywords = keywords_nn + keywords_vb
            # keywords = [item for item in keywords if item in ws_all]
            tmp_kw = keywords[:5]
            tmp_kw = [stemmer.stem(item) for item in tmp_kw]
            keywords_all.append(tmp_kw[:])
            print_idx += 1
    assert len(keywords_all) == len(sum_comment_sentences)
    return keywords_all


def processor(corpus_comments, unique_comment_ids, feat_all):
    """ Do not use semantic_span anymore.
    """
    sum_comment_sentences, sum_cmt_sent_feats, sum_cmt_sent_ids = corpus_comments, feat_all, unique_comment_ids

    keywords_all = list()
    sum_comment_sentences_, sum_cmt_sent_feats_, sum_cmt_sent_ids_ = list(), list(), list()
    if len(sum_comment_sentences) > 0:
        # keywords extraction
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(sum_comment_sentences)
        feature_names = vectorizer.get_feature_names_out()
        dense = vectors.todense()
        dense_list = dense.tolist()
        print_idx = 0
        for tf_idf_scores in dense_list:
            # avoid repeat comments
            if sum_comment_sentences[print_idx] in sum_comment_sentences_:
                print_idx += 1
                continue
            print_all = dict()
            for word, score in zip(feature_names, tf_idf_scores):
                if score > 0:
                    print_all[score] = word
            sorted_scores = sorted(list(print_all.keys()), reverse=True)
            sorted_words = [print_all[score] for score in sorted_scores]
            nn_words, vb_words = list(), list()
            keywords_nn, keywords_vb = list(), list()
            tokens_all = nltk.word_tokenize(sum_comment_sentences[print_idx])
            pos_out = nltk.pos_tag(tokens_all)
            for word_pos in pos_out:
                if word_pos[0] in stopwords.words('english') or word_pos[0] in punctuation:
                    continue
                if word_pos[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
                    nn_words.append(word_pos[0])
                if word_pos[1] in ['VB', 'VBD', 'VBP', 'VBN', 'VBG', 'VBZ']:
                    vb_words.append(word_pos[0])
            for ti_words in sorted_words:
                if ti_words in nn_words:
                    keywords_nn.append(ti_words)
                if ti_words in vb_words:
                    keywords_vb.append(ti_words)
            keywords = keywords_nn + keywords_vb
            
            tmp_kw = keywords[:5]
            tmp_kw = [stemmer.stem(item) for item in tmp_kw]
            keywords_all.append(tmp_kw)
            sum_comment_sentences_.append(sum_comment_sentences[print_idx])
            sum_cmt_sent_feats_.append(sum_cmt_sent_feats[print_idx])
            sum_cmt_sent_ids_.append(sum_cmt_sent_ids[print_idx])
            print_idx += 1
    return sum_comment_sentences_, sum_cmt_sent_feats_, sum_cmt_sent_ids_, keywords_all


def pipeline(args):
    word2vec = dict()
    ucc_lines_all, ucc_count_all = list(), list()
    gen_sum_all = list()
    gen_sum_tt_all = list()

    for article_cluster in args.article_cluster:
        db_data = build_tree_comments_db(article_cluster, args.comment_type, args.w2v_size) if use_db else build_tree_comments(article_cluster, args.comment_type, args.w2v_size)
        comments_all, unique_comment_ids, feat_all, label2id, wv_model = db_data
        semantic_info = processor(comments_all, unique_comment_ids, feat_all)
        sum_comment_sentences_, sum_cmt_sent_feats, sum_cmt_sent_ids, keywords_all = semantic_info

        # build keyword2vec
        for words in keywords_all:
            for word in words:
                if word not in word2vec.keys() and word in wv_model.wv.vocab.keys():
                    word2vec[word] = wv_model.wv[word]
    
        save_data(word2vec, args.w2v_path)
        ac_background = get_background_info(args, article_cluster) if args.bg_icl else None

        # assert len(sum_comment_sentences_) > 0
        # clustering
        data_set = form_set(sum_comment_sentences_, sum_cmt_sent_ids, sum_cmt_sent_feats, label2id, keywords_all, args.comment_type)
        if len(data_set) > 0:
            clustered_ucc, _ = clustering(data_set, args, label2id, ac_background)
        else:
            clustered_ucc = []
        # summarization
        ucc_lines, ucc_line_count = prepare_summarization(clustered_ucc)
        ucc_lines_all = ucc_lines_all + ucc_lines
        ucc_count_all = ucc_count_all + ucc_line_count

        ac_gen_summaries, ac_gen_titles = summary_gen(ucc_lines, ac_background, args.llm_name)
        ac_gen_summaries[-1] = "No summary for other comments."
        gen_sum_all = gen_sum_all + ac_gen_summaries
        gen_sum_tt_all = gen_sum_tt_all + ac_gen_titles
    write_iterate(ucc_lines_all, args.ucc_path)
    save_data(ucc_count_all, args.cached_semantic_group)
    write_iterate(gen_sum_all, output_file)
    write_iterate(gen_sum_tt_all, output_tt_file)


def evaluate_uccs(args, ann_comments_labels=None, ann_comments_all=None, ann_comment_ids_all=None, ann_feats_all=None, ann_sum_all=None):
    word2vec = dict()
    h_scores, nmi_scores, f_scores = 0., 0., 0.
    ucc_lines_all, gold_sum_all, gen_sum_all, gen_sum_tt_all = [], [], [], []
    ucc_count_all = list()
    for ac_id, article_ids in enumerate(args.article_cluster):
        comments_all, comment_ids_all, feat_all, label2id, wv_model = build_eval_comments(ann_comments_all[ac_id],
                                                                                          ann_comment_ids_all[ac_id],
                                                                                          ann_feats_all[ac_id],
                                                                                          args.comment_type, args.w2v_size)
        keywords_all = build_eval_keywords(comments_all)

        # build keyword2vec
        for words in keywords_all:
            for word in words:
                if word not in word2vec.keys() and word in wv_model.wv.vocab.keys():
                    word2vec[word] = wv_model.wv[word]
        save_data(word2vec, args.w2v_path)

        if len(comments_all) > 0:
            # Get background info for a group of articles
            ac_background = get_background_info(args, article_ids, ac_id) if args.bg_icl else None

            # clustering
            # data_set = form_set(comments_all, comment_ids_all, feat_all, label2id, keywords_all, args.comment_type)
            # _, pre_ucc_labels = clustering(data_set, args, label2id, ac_background)
            # gold_uss_labels = ann_comments_labels[ac_id]
            # scores = clustering_score(pre_ucc_labels, gold_uss_labels)
            # nmi_scores += scores[1]

            # summarization
            ucc_lines, gold_sum, ucc_line_count = prepare_eval_summarization(ann_comments_all[ac_id],
                                                                             ann_comment_ids_all[ac_id],
                                                                             ann_sum_all[ac_id], keywords_all)
            ucc_lines_all = ucc_lines_all + ucc_lines
            ucc_count_all = ucc_count_all + ucc_line_count
            gold_sum_all = gold_sum_all + gold_sum
            ac_gen_summaries, ac_gen_titles = summary_gen(ucc_lines, ac_background, args.llm_name)
            gen_sum_all = gen_sum_all + ac_gen_summaries
            gen_sum_tt_all = gen_sum_tt_all + ac_gen_titles
            
    logger.warning("v_measure score of comment clusters: {}".format(nmi_scores / len(args.article_cluster)))
    print("v_measure score of comment clusters: {}".format(nmi_scores / len(args.article_cluster)))

    # summary data
    write_iterate(ucc_lines_all, args.ucc_path)
    save_data(ucc_count_all, args.cached_semantic_group)  # discount & cmt_ids
    write_iterate(gold_sum_all, args.ucc_gold_path)
    write_iterate(gen_sum_all, output_file)
    write_iterate(gen_sum_tt_all, output_tt_file)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--evaluate", action="store_true", help="Evaluate the two-stage system or not.")
    arg_parser.add_argument("--eval_pipeline", action="store_true", help="Evaluate the pipeline system or not.")
    arg_parser.add_argument("--dev", action="store_true", help="Evaluate the two-stage system or not.")

    arg_parser.add_argument("--article_cluster", default=None)
    arg_parser.add_argument("--cversion", default=1, type=int)
    arg_parser.add_argument("--cword_rep", default="sentence-transformer")  # glove, transformer, sentence-transformer
    arg_parser.add_argument("--comment_type", default="cmt_list")  # cmt_list or cmt_tree
    arg_parser.add_argument("--cfeature_space", default=2, type=int)
    arg_parser.add_argument("--cfeature_label", default=3, type=int)
    arg_parser.add_argument("--cmt_id_feature", action="store_true", help="Indicate whether two semantic spans belong to one comment or not.")
    arg_parser.add_argument("--ckw_feature", action="store_true", help="Whether use the representation of keywords for clustering or not.")
    arg_parser.add_argument("--semantic_span", action="store_true", help="Combine continuous sentences inside a comment as semantic spans.")
    arg_parser.add_argument("--dynamic_similar_thr", action="store_true", help="Use dynamic similarity or not.")
    arg_parser.add_argument("--similar_max_min", default=None) 
    arg_parser.add_argument("--hyper", default=0.8, type=float) 
    arg_parser.add_argument("--w2v_path", default=cached_w2v)
    arg_parser.add_argument("--w2v_size", default=64, type=int)
    # Summary attributions
    arg_parser.add_argument("--ucc_path", default=eval_article_path)
    arg_parser.add_argument("--ucc_gold_path", default=eval_summary_path)
    arg_parser.add_argument("--cached_semantic_group", default=cached_semantic_group)
    arg_parser.add_argument("--cached_ac_path", default=cached_ac_ids)
    # evaluation attributions
    arg_parser.add_argument("--manual_ann_path", default=manual_test_path)
    arg_parser.add_argument("--use_db", default=use_db)
    arg_parser.add_argument("--sent_rep_name", default="all-MiniLM-L6-v2")
    arg_parser.add_argument("--llm_name", default="output_uccs_cot_title/checkpoint-8679")
    arg_parser.add_argument("--bg_icl", action="store_true", help="Whether use the BG info for in-context-learning or not.")
    arg_parser.add_argument("--llm_bg", action="store_true", help="Whether use the summarized BG.")
    arg_parser.set_defaults(use_gpu=True)

    arg_parser.add_argument("--km_val", default=1, type=int)

    args_ = arg_parser.parse_args()

    args_.similar_max_min = [0.54, 0.95, 0.8, 200]
    print("min similarity value: ", args_.similar_max_min)

    if args_.dev:
        # For hyper-parameter selection 
        args_.manual_ann_path = manual_dev_path

    if args_.evaluate:
        args_.use_db = False
        # Load data for evaluation
        ac_article_ids, comments_all_, comment_ids_, comments_feat_all_, comments_labels_, sum_all_ = parse_eval_data(args_.manual_ann_path)
        args_.article_cluster = ac_article_ids
        evaluate_uccs(args_, comments_labels_, comments_all_, comment_ids_, comments_feat_all_, sum_all_)
        logger.info("Clustering evaluate done.")
        torch.cuda.empty_cache()
        exit()

    if args_.eval_pipeline:
        args_.use_db = False
        ac_article_ids, comments_all_, _, _, _, sum_all_ = parse_eval_data(args_.manual_ann_path)
        args_.article_cluster = ac_article_ids

        # gold summaries
        gold_sum_all = []
        for ac_id, _ in enumerate(ac_article_ids):
            gold_sum_all += [item.strip() for item in sum_all_[ac_id]]
        write_iterate(gold_sum_all, args_.ucc_gold_path)
        pipeline(args_)
        torch.cuda.empty_cache()
        exit()

    # Last situation is made for real-senario test.
    if args_.use_db:
        ac_article_ids, ac_id_list = db_obj.get_ac_ids()
        save_data((ac_article_ids, ac_id_list), args_.cached_ac_path)
    else:
        with open(carticle_clusters) as f:
            reader = csv.DictReader(f)
            ac_article_ids = [item["article_ids"].split() for item in reader]
    args_.article_cluster = ac_article_ids
    pipeline(args_)
    logger.info("Clustering done.")
