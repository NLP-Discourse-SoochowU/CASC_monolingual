#!/bin/sh

# final settings
TF_ENABLE_ONEDNN_OPTS=0 python3 run_ucc.py --evaluate --llm_name ../../flan_t5_xl/output_uccs_cot/checkpoint-4770 --cversion 1 --ckw_feature --w2v_size 128 --bg_icl
TF_ENABLE_ONEDNN_OPTS=0 python3 postprocess.py --use_at --evaluate

# ablation on clustering
# TF_ENABLE_ONEDNN_OPTS=0 python3 run_ucc.py --evaluate --llm_name ../../flan_t5_xl/output_uccs_cot/checkpoint-4770 --cversion 1 --w2v_size 128 --dev
# TF_ENABLE_ONEDNN_OPTS=0 python3 postprocess.py --use_at --evaluate --dev

# ablation on summary
# for iter in 2;
# do
# TF_ENABLE_ONEDNN_OPTS=0 python3 run_ucc.py --evaluate --llm_name ../../flan_t5_xl/output_uccs_cot/checkpoint-4770 --cversion 1 --ckw_feature --w2v_size 128 --bg_icl --km_val $iter --dev
# TF_ENABLE_ONEDNN_OPTS=0 python3 postprocess.py --use_at --evaluate --dev
# done

# clustering HPs
# --hyper $iter
# --hyper $hyperv --dev --bg_icl --llm_bg 
# --cmt_id_feature --cfeature_space 2 
# --ckw_feature --w2v_size 64
# --sent_rep_name hkunlp/instructor-large | ../../instructor/output_v2

# summary HPs
# --bg_icl
# google/flan-t5-xl  ->  the vanilla FLT model
# ../../flan_t5_xl/output_uccs/checkpoint-3315  -> tuned basically
# ../../flan_t5_xl/output_uccs_cot/checkpoint-4770  ->  COT tuning with AT generation;
# ../../flan_t5_xl/output_uccs_cot_tt/checkpoint-8679  ->  COT tuning with title generation;

# # affinity p  265ms for 512 cmts
# TF_ENABLE_ONEDNN_OPTS=0 python3 run_ucc.py --evaluate --llm_name ../../flan_t5_xl/output_uccs_cot/checkpoint-4770 --cversion 2 --ckw_feature --w2v_size 128 --bg_icl

# k-means K  228ms for 512 cmts || fast clustering comments 512: 86ms
# TF_ENABLE_ONEDNN_OPTS=0 python3 run_ucc.py --evaluate --llm_name ../../flan_t5_xl/output_uccs_cot/checkpoint-4770 --cversion 3 --ckw_feature --w2v_size 128 --bg_icl

# GPT performance BAD
# TF_ENABLE_ONEDNN_OPTS=0 python3 run_ucc.py --evaluate --llm_name gpt --cversion 1 --ckw_feature --w2v_size 128 --bg_icl
# TF_ENABLE_ONEDNN_OPTS=0 python3 postprocess.py --use_at --evaluate

# for iter in 8;
# do
# TF_ENABLE_ONEDNN_OPTS=0 python3 run_ucc.py --evaluate --llm_name ../../flan_t5_xl/output_uccs_cot/checkpoint-4770 --cversion 3 --ckw_feature --w2v_size 128 --bg_icl
# --km_val $iter
# done
# 
# Test the original FC speed
