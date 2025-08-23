from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from typing import List, Tuple
import os

word_on="ON"
word_off="OFF"

class LLMHandler:
    def __init__(self, model_name: str, task_type: str = "classification"):
        self.model_name = model_name
        self.task_type = task_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_8bit=False,
            load_in_4bit=False,
        )
        self.token_ids = self._resolve_token_ids(model_name)

    def generate_prediction(self, prompt: str) -> Tuple[str, List[float]]:
        """Return decoded prediction and ON/OFF token probabilities."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        self._print_transtion_score(out, inputs)

        decoded = self.tokenizer.decode(out.sequences[0], skip_special_tokens=True)
        scores = out.scores[0]     
        prob, norm_prob = self._extract_token_probs(scores)
        print(f"prob: {prob}")
        print(f"token_ids: {self.token_ids}")
        return decoded.strip(), prob, norm_prob

    def _print_transtion_score(self, output, inputs):
        # Retrieve the raw token logit for ON and OFF
        transition_scores = self.model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = output.sequences[:, input_length:]
        for tok, score in zip(generated_tokens[0], transition_scores[0]):
            print(f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.cpu().numpy():.3f} | {np.exp(score.cpu().numpy()):.2%}")

    def _resolve_token_ids(self, name: str) -> np.ndarray:
        ids_on  = self.tokenizer.encode(word_on,
            add_special_tokens=False,)
        ids_off  = self.tokenizer.encode(word_off,
                                        add_special_tokens=False,)
        print(ids_on)
        print(ids_off)
        return [ids_on[0], ids_off[0]] #works only for ON and OFF, case sensitive!!
    
    def _extract_token_probs(self, logits: torch.Tensor) -> List[float]:
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        raw_probs = [probs[token_id].item() for token_id in self.token_ids]
    
        binary_logits = logits[0][self.token_ids]
        normalized_probs = torch.nn.functional.softmax(binary_logits, dim=0).tolist()
        return raw_probs, normalized_probs
