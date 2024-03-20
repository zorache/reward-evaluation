import os
os.environ['HF_HOME'] = '/cmlscratch/zche/.cache/huggingface'
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download


import math
reward_device = "cuda"
reward_batch_size = 64

## Define the reward model function class

class GPTRewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_path)
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.model = model
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        """
        input_ids, attention_mask: torch.Size([bs, seq_len])
        return: scores: List[bs]
        """
        bs = input_ids.shape[0]
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])
        return scores

## Load the model and tokenizer

# reward_model = GPTRewardModel("meta-llama/Llama-2-7b-chat-hf")
# reward_tokenizer = reward_model.tokenizer
# reward_tokenizer.truncation_side = "left"


# directory = snapshot_download("berkeley-nest/Starling-RM-7B-alpha")
# for fpath in os.listdir(directory):
#     if fpath.endswith(".pt") or fpath.endswith("model.bin"):
#         checkpoint = os.path.join(directory, fpath)
#         break
   
# reward_model.load_state_dict(torch.load(checkpoint), strict=False)
# reward_model.eval().requires_grad_(False)


# ## Define the reward function

# def get_reward(samples):
#     """samples: List[str]"""
#     input_ids = []
#     attention_masks = []
#     encodings_dict = reward_tokenizer(
#         # samples,
#         samples,
#         truncation=True,
#         max_length=256,
#         padding="max_length",
#         return_tensors="pt",
#     )
#     # ).to(reward_device)

#     input_ids = encodings_dict["input_ids"]
#     attention_masks = encodings_dict["attention_mask"]
#     mbs = reward_batch_size
#     out = []
#     for i in range(math.ceil(len(samples) / mbs)):
#         rewards = reward_model(input_ids=input_ids[i * mbs : (i + 1) * mbs], attention_mask=attention_masks[i * mbs : (i + 1) * mbs])
#         out.extend(rewards)
#     return torch.hstack(out)

# ## Inference over test prompts with llama2 chat template
# # Does the chat format matter?

# test_sample = ["<s>[INST] Hello? </s> [/INST] Hi, how can I help you?</s>"] 

# reward_for_test_sample = get_reward(test_sample)
# print(reward_for_test_sample)


# test_sample = ["<s>[INST] Hello? </s> [/INST] Shut up</s>"] 

# reward_for_test_sample = get_reward(test_sample)
# print(reward_for_test_sample)


# B_INST, E_INST = "[INST]", "[/INST]"
# B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# SYS_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

# # PROMPT_TEMPLATE = B_INST+ B_SYS + SYS_PROMPT + E_SYS + "%s" + E_INST

# WRAPPED_PROMPT = B_INST + " " + B_SYS + SYS_PROMPT + E_SYS + "%s" + " " + E_INST


# template = [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': '{instruction}'}] if system_message else [{'role': 'user', 'content': '{instruction}'}]
# prompt = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)