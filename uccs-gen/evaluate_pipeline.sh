# TF_ENABLE_ONEDNN_OPTS=0 python3 run_ucc.py --llm_name google/flan-t5-xl --cversion 1 --ckw_feature --w2v_size 128 --eval_pipeline
# TF_ENABLE_ONEDNN_OPTS=0 python3 postprocess.py --use_at --eval_pipeline

# TF_ENABLE_ONEDNN_OPTS=0 python3 run_ucc.py --llm_name ../../flan_t5_xl/output_uccs_cot/checkpoint-4770 --cversion 1 --ckw_feature --w2v_size 128 --bg_icl --eval_pipeline
# TF_ENABLE_ONEDNN_OPTS=0 python3 postprocess.py --use_at --eval_pipeline

# TF_ENABLE_ONEDNN_OPTS=0 python3 run_ucc.py --llm_name ../../flan_t5_xl/output_uccs_cot/checkpoint-4770 --cversion 4 --ckw_feature --w2v_size 128 --bg_icl --eval_pipeline
# TF_ENABLE_ONEDNN_OPTS=0 python3 postprocess.py --use_at --eval_pipeline

# Dynamic xxx, the change was made in the file train v1
# TF_ENABLE_ONEDNN_OPTS=0 python3 run_ucc.py --llm_name ../../flan_t5_xl/output_uccs_cot/checkpoint-4770 --cversion 1 --ckw_feature --w2v_size 128 --bg_icl --eval_pipeline
# TF_ENABLE_ONEDNN_OPTS=0 python3 postprocess.py --use_at --eval_pipeline

# LLaMa
TF_ENABLE_ONEDNN_OPTS=0 python3 run_ucc.py --llm_name llama --cversion 1 --ckw_feature --w2v_size 128 --bg_icl --eval_pipeline
TF_ENABLE_ONEDNN_OPTS=0 python3 postprocess.py --use_at --eval_pipeline

# GPT
# TF_ENABLE_ONEDNN_OPTS=0 python3 run_ucc.py --llm_name gpt --cversion 1 --ckw_feature --w2v_size 128 --bg_icl --eval_pipeline
# TF_ENABLE_ONEDNN_OPTS=0 python3 postprocess.py --use_at --eval_pipeline

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

# LLaMa 0.8
# Hit Rate: 0.48775510204081635-P, 0.3414285714285714-R, 0.4016806722689075-F1

# LLaMa + fine_tune
# Hit Rate: 0.4714587737843552-P, 0.31857142857142856-R, 0.3802216538789429-F1

# LLaMa2 meta-llama/Llama-2-7b-chat-hf
# Hit Rate: 0.7632135306553911-P, 0.5157142857142857-R, 0.6155157715260017-F1

# GPT 
# 0.9 DyFC Evaluation 
# Hit Rate: 0.7917525773195876-P, 0.5485714285714286-R, 0.6481012658227848-F1