import os
import sys
import time
from string import punctuation
import random
import torch
from peft import PeftModel
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from util.file_util import *
import nltk
from model.static_parameters import *
from util.file_util import get_stem, save_data
sys.path.append("..")
from app_cfg import *
from threading import Thread
from gpt import add_test_paper
import progressbar

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

def random_prompts(cc_list, background_info, background_stem, total_token_max=512):
    diverse_at = [
        "Retrieve the primary aspect words deliberated upon among the provided user comments.",
        "Identify the key aspects discussed within the following user comments.",
        "Extract the central aspect terms appear in the given user comments.",
        "Ascertain the principal aspects discussed in the following user comments.",
        "Distill the primary aspects expressed within the given user comments.",
        "Capture the essential aspect terms talked about between the following user comments.",
        "Unearth the primary aspect words analyzed in the provided user comments.",
        "Summarize the main aspects discussed within the following user comments."
    ]
    diverse_instructions = [
        "incorporate information from the surrounding sentences in a user comment cluster to provide context and generate an informative summary.",
        "explore generating a summary that is fluent and coherent.",
        "generate a concise summary for user comments while preserving important information.",
        "consider the length of the input user comment cluster and adjust the summarization approach accordingly, ensuring that the summary is appropriately concise or detailed.",
        "explore methods of handling contradictory or conflicting information within a user comment cluster to generate a summary that captures different perspectives or uncertainties.",
        "identify the most representative sentences in a user comment cluster and use them as the summary for that cluster.",
        "determine the key sentences within a user comment cluster and generate a summary based on them.",
        "consider the semantic relationships between sentences in a user comment cluster and generate a summary that captures the main ideas.",
        "focus on identifying and summarizing the most important information or events mentioned in the cluster of user comments.",
        "incorporate information from the surrounding sentences in a user comment cluster to provide context and generate an informative summary.",
        "generate a concise summary for a group of user comments that share a common theme or topic."
    ]
    prompt_made_all, prompt_count_all = list(), list()
    for cc_one in cc_list:
        cc_one = cc_one[:100]
        cc_word_stm_set = get_stem(" ".join(cc_one))
        cc_one_str_list = [f"User Comment {item_id + 1}. " + item.strip() + " " for item_id, item in enumerate(cc_one)]
        cc_one_str = ""
        token_num_count = 0
        while token_num_count < total_token_max and len(cc_one_str_list) > 0:
            next_str = cc_one_str_list.pop(0)
            next_str_len = len(next_str.split())
            if token_num_count + next_str_len <= total_token_max or cc_one_str == "":
                token_num_count += next_str_len
                cc_one_str = cc_one_str + " " + next_str

        prompt_count_all.append(token_num_count)

        prompt_made = random.sample(diverse_at, 1)[0] + " Based on these aspects, " + random.sample(diverse_instructions, 1)[0] + " The user comment cluster is as follows: " + cc_one_str + "\nTwo rows should be returned, the first with less than 5 aspect terms and the second with the summary."

        # add background
        if background_info is not None:
            background_info = background_info.strip()
            if len(background_info) > 0 and len(cc_word_stm_set & background_stem) > 1:
                prompt_made = f"Given the following news background '{background_info}', there are some user comments about the news. " + prompt_made
        prompt_made_all.append(prompt_made)
    return prompt_made_all, prompt_count_all

def random_prompts_tt(cc_list, background_info=None, background_stem=None, total_token_max=512):
    diverse_instructions = [
        "incorporate information from the surrounding sentences in a user comment cluster to provide context and generate an informative summary.", 
        "explore generating a summary that is fluent and coherent.", 
        "generate a concise summary for user comments while preserving important information.", 
        "consider the length of the input user comment cluster and adjust the summarization approach accordingly, ensuring that the summary is appropriately concise or detailed.", 
        "explore methods of handling contradictory or conflicting information within a user comment cluster to generate a summary that captures different perspectives or uncertainties.", 
        "identify the most representative sentences in a user comment cluster and use them as the summary for that cluster.", 
        "determine the key sentences within a user comment cluster and generate a summary based on them.", 
        "consider the semantic relationships between sentences in a user comment cluster and generate a summary that captures the main ideas.", 
        "focus on identifying and summarizing the most important information or events mentioned in the cluster of user comments.", 
        "incorporate information from the surrounding sentences in a user comment cluster to provide context and generate an informative summary.", 
        "generate a concise summary for a group of user comments that share a common theme or topic."
    ]
    prompt_made_all, prompt_count_all = list(), list()
    for cc_one in cc_list:
        cc_one = cc_one[:100]
        cc_word_stm_set = get_stem(" ".join(cc_one))
        cc_one_str_list = [f"User Comment {item_id + 1}. " + item.replace("bm $$$$$$ ", "").replace("bi $$$$$$ ", "").replace("cn $$$$$$ ", "").replace("en $$$$$$ ", "").strip() + " " for item_id, item in enumerate(cc_one)]
        cc_one_str = ""
        token_num_count = 0
        while token_num_count < total_token_max and len(cc_one_str_list) > 0:
            next_str = cc_one_str_list.pop(0)
            next_str_len = len(next_str.split())
            if token_num_count + next_str_len <= total_token_max or cc_one_str == "":
                token_num_count += next_str_len
                cc_one_str = cc_one_str + " " + next_str

        prompt_count_all.append(token_num_count)

        # prompt_made = "Please " + random.sample(diverse_instructions, 1)[0] + "The user comment cluster is as follows: " + cc_one_str
        prompt_made = "Please " + random.sample(diverse_instructions, 1)[0] + "Besides, generate a short title for the obtained summary. The user comment cluster is as follows: " + cc_one_str + "\nTwo rows should be returned, the first with the summary and the second with the short title."

        # add background
        if background_info is not None:
            background_info = background_info.strip()
            if len(background_info) > 0 and len(cc_word_stm_set & background_stem) > 1:
                prompt_made = f"Given the following news background '{background_info}', there are some user comments about the news. " + prompt_made
        prompt_made_all.append(prompt_made)
    return prompt_made_all, prompt_count_all


def random_prompts_add(cc_list, total_token_max=512):
    prompt_made_all, prompt_count_all = list(), list()
    for cc_one in cc_list:
        cc_one = cc_one[:100]
        cc_word_stm_set = get_stem(" ".join(cc_one))
        cc_one_str_list = [item.strip() for item in cc_one]
        cc_one_str = ""
        token_num_count = 0
        while token_num_count < total_token_max and len(cc_one_str_list) > 0:
            next_str = cc_one_str_list.pop(0)
            next_str_len = len(next_str.split())
            if token_num_count + next_str_len <= total_token_max or cc_one_str == "":
                token_num_count += next_str_len
                cc_one_str = cc_one_str + " " + next_str
        prompt_count_all.append(token_num_count)
        prompt_made = "[INST] Please summarize the following user comments: " + cc_one_str + "[/INST]"
        prompt_made_all.append(prompt_made)
    return prompt_made_all, prompt_count_all


def one_thread_ucs(group_prompts, group_cc_ids, summary_dict, cuda_id, model_):
    with torch.no_grad():
        for batch_prompt, batch_cc_ids in zip(group_prompts, group_cc_ids):
            pass_flag = False 
            inputs = tokenizer(batch_prompt, return_tensors="pt", padding=True)
            cc_ids = inputs["input_ids"].to(f"cuda:{cuda_id}")
            while not pass_flag:
                try:
                    cc_summaries = model_.generate(cc_ids, max_new_tokens=1000)
                    cc_summaries = tokenizer.batch_decode(cc_summaries)
                    pass_flag = True
                except:
                    time.sleep(1)
                    print("Re-trying...")
            cc_summaries = [item.replace("<pad>", "").replace("</s>", "").replace("END</s>", "").strip() for item in cc_summaries]
            for cc_id, summary_one in zip(batch_cc_ids, cc_summaries):
                summary_dict[cc_id] = summary_one

def split_list(ori_list, n):
    group_prompts = [[] for _ in range(n)]
    for item_idx, item in enumerate(ori_list):
        group_id = item_idx % n
        group_prompts[group_id].append(item)
    return group_prompts

def build_data_batch(prompt_all_num, prompt_all, prompt_count_all, cc_list, total_token_max):
    batch_idx = 0
    batch_prompt_all, batch_cc_id_all, batch_cc_all = list(), list(), list()
    while batch_idx < prompt_all_num:
        max_len, token_num, batch_size = 0, 0, 1
        batch_prompt, batch_cc_id, batch_cc = list(), list(), list()

        while token_num < total_token_max and batch_idx < prompt_all_num:
            max_len = max(max_len, prompt_count_all[batch_idx]) 
            if len(batch_prompt) != 0 and max_len * batch_size > total_token_max:
                break
            token_num = max_len * batch_size
            # else do something
            batch_prompt.append(prompt_all[batch_idx])
            batch_size += 1
            batch_cc_id.append(batch_idx)
            batch_cc.append(cc_list[batch_idx])
            batch_idx += 1
            if batch_idx == 1:
                break  # The first one is too big, do not suggest batch computing
        batch_prompt_all.append(batch_prompt)
        batch_cc_id_all.append(batch_cc_id)
        batch_cc_all.append(batch_cc)
    return batch_prompt_all, batch_cc_id_all, batch_cc_all

def summary_gen(ucc_lines_all, background_info, llm_path):
    models = list()
    for cuda_id in cuda_cards_buffer:
        model = AutoModelForSeq2SeqLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16, use_cache=False)
        model = model.to(f"cuda:{cuda_id}")
        models.append(model)
    
    background_stem = None if background_info is None else get_stem(background_info)
    cc_list = [line.strip().split(" <S_SEP> ") for line in ucc_lines_all if len(line.strip()) > 0]
    
    cc_summaries_all = list()
    prompt_all, prompt_count_all = random_prompts_tt(cc_list, background_info, background_stem)
    prompt_all_num = len(prompt_all)
    
    batch_prompt_all, batch_cc_id_all, batch_cc_all = build_data_batch(prompt_all_num, prompt_all, prompt_count_all, cc_list, total_token_max=llm_input_limit)

    cc_summaries_all = list()
    # calculate the computing resources
    group_num = len(cuda_cards_buffer)
    batch_prompt_groups = split_list(batch_prompt_all, group_num * thread_num_per_gpu)
    batch_cc_id_groups = split_list(batch_cc_id_all, group_num * thread_num_per_gpu)
    try:
        group_summaries = dict()
        threads_all = list()
        for group_id, cuda_id in enumerate(cuda_cards_buffer):
            for _ in range(thread_num_per_gpu):
                one_group_prompts = batch_prompt_groups.pop(0)
                one_group_cc_ids = batch_cc_id_groups.pop(0)
                thread_one = Thread(target=one_thread_ucs, args=(one_group_prompts, one_group_cc_ids, group_summaries, cuda_id, models[group_id]))
                thread_one.start()
                threads_all.append(thread_one)
        # wait for all to be done
        for thread_one in threads_all:
            thread_one.join()
    except:
        print("Error: unable to start thread")

    summary_all, summary_titles = list(), list()
    for batch_cc_id, batch_cc in zip(batch_cc_id_all, batch_cc_all):
        for cc_id, cc in zip(batch_cc_id, batch_cc):
            if cc_id not in group_summaries.keys():
                continue
            cc_summary = group_summaries[cc_id].strip()
            cc_title = "No title."
            if "Title:" in cc_summary:
                cc_sum_parts = cc_summary.split("Title:")
                cc_sum = "Title:".join(cc_sum_parts[:-1])
                cc_title = cc_sum_parts[-1]

                # post process the summary
                fragments = cc_sum.split(". ")
                if len(fragments) > 1 and cc_sum[-1] not in punctuation:
                    if len(fragments[-1].split()) <= 5:
                        # throw away
                        fragments = fragments[:-1]
                
                cc_sent_num = 0 
                for cc_item in cc:
                    cc_sent_num += len(cc_item.split(". "))
                fragments = fragments[:cc_sent_num]  # control the summary length
                
                cc_sum = ". ".join(fragments).strip()
                if cc_sum[-1] not in punctuation:
                    cc_sum = cc_sum  + "."
            else:
                cc_sum = cc_summary
            summary_titles.append(cc_title)
            summary_all.append(cc_sum)
    return summary_all, summary_titles

# def summary_gen(ucc_lines_all, background_info, llm_path):
#     cc_list = [line.strip().split(" <S_SEP> ") for line in ucc_lines_all if len(line.strip()) > 0]

#     summary_all, summary_titles = list(), list()
#     if llm_path == "gpt":
#         prompt_all, _ = random_prompts_add(cc_list)
#         summary_all, summary_titles = add_test_paper(prompt_all)
#     else:
#         global tokenizer 
#         tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir="new_pre_trained_files/llama/")
#         model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16, cache_dir="new_pre_trained_files/llama/").to("cuda")
#         model = PeftModel.from_pretrained(model, "../../alpaca-lora/lora-llama2", torch_dtype=torch.bfloat16).to("cuda")
#         # tokenizer = LlamaTokenizer.from_pretrained("../../alpaca-lora/hf_ckpt", cache_dir="new_pre_trained_files/llama/")
#         # model = LlamaForCausalLM.from_pretrained("../../alpaca-lora/hf_ckpt", torch_dtype=torch.bfloat16, cache_dir="new_pre_trained_files/llama/").to("cuda")
#         # model = PeftModel.from_pretrained(model, "../../alpaca-lora/lora-alpaca-uccs-v5", torch_dtype=torch.bfloat16).to("cuda")
#         model.eval()

#         prompt_all, _ = random_prompts_add(cc_list)
#         p = progressbar.ProgressBar()
#         p.start(len(prompt_all))
#         p_value = 0
#         for pmt in prompt_all:
#             # print("===================================")
#             # print(pmt)
#             # print()
#             inputs = tokenizer(pmt, return_tensors="pt")
#             input_ids = inputs["input_ids"].to("cuda")

#             generation_config = GenerationConfig(
#                 temperature=0.1,
#                 top_p=0.85,
#                 top_k=20,
#                 num_beams=2,
#                 do_sample=True
#             )

#             with torch.no_grad():
#                 try:
#                     generation_output = model.generate(
#                         input_ids=input_ids,
#                         generation_config=generation_config,
#                         return_dict_in_generate=True,
#                         output_scores=True,
#                         max_new_tokens=119,
#                     )
#                     s = generation_output.sequences[0]
#                     output = tokenizer.decode(s)
#                     output = output.replace("</s>", "").replace("<s>", "").strip()
#                     response_str = output.split("[/INST]")[-1].strip().replace("\n", "")  # For LLaMa2
#                     # response_str = output.replace(pmt, "").strip().replace("\n", "")  # For LLaMa
#                 except:
#                     response_str = "NULL"
#             summary_all.append(response_str)
#             summary_titles.append("")
#             p_value += 1
#             p.update(p_value)
#         p.finish()
#     return summary_all, summary_titles

def title_sum_gen(titles, llm_path, total_token_max=512):
    if len(titles) <= 3:
        return " ".join(titles)
    model = AutoModelForSeq2SeqLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16, use_cache=False)
    model = model.to(f"cuda:{cuda_cards_buffer[0]}")

    titles_str = ["Summarize for the following titles."] + [f"\nTitle {idx + 1}: {item}" for idx, item in enumerate(titles)]
    prompt_made, token_num_count = "", 0
    while token_num_count < total_token_max and len(titles_str) > 0:
        next_str = titles_str.pop(0)
        next_str_len = len(next_str.split())
        if token_num_count + next_str_len <= total_token_max or cc_one_str == "":
            token_num_count += next_str_len
            prompt_made = prompt_made + " " + next_str
    
    inputs = tokenizer(prompt_made, return_tensors="pt", padding=True)
    tt_ids = inputs["input_ids"].to(f"cuda:{cuda_cards_buffer[0]}")
    try:
        tt_summaries = model.generate(tt_ids, max_new_tokens=1000)
        tt_summaries = tokenizer.decode(tt_summaries[0])
        tt_summaries = tt_summaries.replace("<pad>", "").replace("</s>", "").replace("END</s>", "").strip()
    except:
        tt_summaries = None
    if tt_summaries.startswith("Summary: "):
        tt_summaries = tt_summaries.replace("Summary: ", "")
    return tt_summaries
