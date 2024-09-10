from transformers import AutoTokenizer, AutoModel
import torch

# class CurseEncoder:
#     def __init__(self, enc_id, hf_token, device='cuda:0'):
#         self.tokenizer = AutoTokenizer.from_pretrained(enc_id, token=hf_token)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.model = AutoModel.from_pretrained(enc_id, token=hf_token).to(device)
#         self.device = device

#     def encode(self, text):
#         inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         with torch.no_grad():
#             outputs = self.model(**inputs, output_hidden_states=True)
#         last_hidden_state = outputs.hidden_states[-1]
#         feature_tensor, _ = torch.max(last_hidden_state, dim=1)
        
#         return feature_tensor
    
#     def free_llm(self):
#         del self.tokenizer
#         del self.model

class CurseEncoder():
    def __init__(self, enc_id, hf_token, device='cuda:0'):
        self.tokenizer = AutoTokenizer.from_pretrained(enc_id, token=hf_token)
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.tokenizer.padding_side = "right"
        torch_dtype = 'auto' if torch.cuda.is_available() else torch.float32
        self.llm = AutoModel.from_pretrained(enc_id, token=hf_token).to(device)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        for name, tensor in self.llm.named_parameters():
            tensor.requires_grad = False
        self.device = device

    def encode(self, curse):
        inputs = self.tokenizer(curse, padding=True, return_tensors='pt').to(self.device)
        sequence_lengths = (torch.eq(inputs.input_ids, self.llm.config.pad_token_id).long().argmax(-1)-1).to(self.device)
        transformer_outputs = self.llm(**inputs)
        hidden_states = transformer_outputs[0]
        logits = hidden_states[torch.arange(1, device=self.device), sequence_lengths]

        return logits
    
    def free_llm(self):
        del self.tokenizer
        del self.model
