# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_equal_to_4_46
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING: 
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments

from transformers import AutoModelForCausalLM

logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], model_args=None, data_args=None, pretrain_model=None, **kwargs   # 字符串表示类型，增强静态检查的准确性
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.model_args = model_args
        self.data_args = data_args
        self.pretrain_model = pretrain_model
        # self.ref_model = AutoModelForCausalLM.from_pretrained("/hujinwu/LLM_Assemble/pretrain_model/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16).to("cuda")

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            labels = labels.detach().clone() if labels is not None else None  # backup labels
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "a", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res)+"\n")


    
    @torch.no_grad()
    def cal_ce(self, logits, labels):
        """
        计算 cross entropy
        """
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits: "torch.Tensor" = logits[..., :-1, :]
        shift_labels: "torch.Tensor" = labels[..., 1:]
        loss_mask = shift_labels != IGNORE_INDEX
        flatten_logits = shift_logits.contiguous().view(shift_labels.size(0)*shift_labels.size(1), -1)
        flatten_labels = shift_labels.contiguous().view(-1)
        token_logps: "torch.Tensor" = criterion(flatten_logits, flatten_labels) # [bs*seq_len]
        token_logps = token_logps.contiguous().view(shift_logits.size(0), -1)  # [bs, seq_len]
        sentence_logps_normal = (token_logps*loss_mask).sum(-1) / loss_mask.sum(-1)  # [bs]
        return sentence_logps_normal
    

    def cal_kl(self, logits, labels):
        loss_fct = nn.KLDivLoss(reduction='batchmean') 

        shift_logits: "torch.Tensor" = logits[..., :-1, :]
        shift_labels: "torch.Tensor" = labels[..., 1:]

        sentence_kl = torch.zeros(shift_logits.size(0), device=shift_logits.device)  # [bs]
        for i, (shift_logit, shift_label) in enumerate(zip(shift_logits, shift_labels)):
            mask = shift_label != IGNORE_INDEX
            shift_logit, shift_label = shift_logit[mask], shift_label[mask]
            log_probs = shift_logit.log_softmax(dim=-1)
            one_hot_targets = torch.zeros_like(log_probs).scatter_(1, shift_label.unsqueeze(1), 1).to(log_probs.device)
            sentence_kl[i] = loss_fct(log_probs, one_hot_targets) 

        return sentence_kl


    def log_to_file(self, sentence_ce, sentence_kl, mask, coeff, loss):
        """
        Log the sentence cross-entropy and KL divergence to a file.
        """
        with open(f"{self.args.output_dir}/logfile.txt", 'a', encoding="utf-8") as f:
            for ce, kl, m, coef in zip(sentence_ce.clone().detach(), sentence_kl.clone().detach(), mask, coeff):
                if m:
                    print(f"This sample is selected. Threshold: {self.finetuning_args.threshold}, Cross-entropy: {ce}, KL divergence: {kl}, Weight coefficient: {coef}, Final loss: {loss}", file=f)
                else:
                    print(f"This sample is discarded. Threshold: {self.finetuning_args.threshold}, Cross-entropy: {ce}, KL divergence: {kl}", file=f)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        assert self.tokenizer.padding_side == 'right', "Training should be done with right padding."
        if self.finetuning_args.setting == "offline_ttl":
            
            # 1. In offline setting, perform a forward pass using the base model to get logits
            with torch.no_grad(): 
                model.eval()
                with self.accelerator.unwrap_model(model).disable_adapter():
                    pretrain_logits = model(**inputs).logits
                    sentence_ce = self.cal_ce(pretrain_logits, inputs["labels"])

            # 2. Filter samples based on cross-entropy (equivalent to KL divergence), keep those above threshold, and calculate weighting coefficients
            mask = sentence_ce > self.finetuning_args.threshold 
            coeff = self.finetuning_args.lamb * torch.exp(sentence_ce.clone().detach() - self.finetuning_args.threshold) # [bs,]
            
            model.train() # Resume training mode
            outputs = model(**inputs)  # Forward pass to get logits
            # 3. Calculate KL divergence
            sentence_kl = self.cal_kl(outputs.logits, inputs["labels"])
            
            # 4. Compute total loss
            sentence_kl = sentence_kl.mul(coeff).mul(mask)  # [bs,]
            if mask.sum() == 0:
                total_loss = sentence_kl.mean()
            else:
                total_loss = sentence_kl.sum() / mask.sum()

            self.log_to_file(sentence_ce.clone().detach(), sentence_kl.clone().detach(), mask, coeff, total_loss.item())

        elif self.finetuning_args.setting == "online_ttl":
            # 1. Perform a forward pass using the model being trained to get logits
            outputs = model(**inputs) 
            # 2. Filter samples based on cross-entropy; in online setting, CE is calculated from current model 
            sentence_ce = self.cal_ce(outputs.logits, inputs["labels"])  # [bs,]
            mask = sentence_ce > self.finetuning_args.threshold  # Keep samples above threshold
            # 3. Calculate KL divergence
            sentence_kl = self.cal_kl(outputs.logits, inputs["labels"])  # [bs,]
            coeff = self.finetuning_args.lamb * torch.exp(sentence_ce.clone().detach() - self.finetuning_args.threshold) # [bs,]
            
            # 4. Compute total loss
            sentence_kl = sentence_kl.mul(coeff).mul(mask)  # [bs,]

            if mask.sum() == 0:
                total_loss = sentence_kl.mean()
            else:
                total_loss = sentence_kl.sum() / mask.sum()

            self.log_to_file(sentence_ce.clone().detach(), sentence_kl.clone().detach(), mask, coeff, total_loss.item())
        
        
        if is_transformers_version_equal_to_4_46() and not getattr(self, "model_accepts_loss_kwargs", False):
            # other model should not scale the loss
            if return_outputs:
                return(total_loss / self.args.gradient_accumulation_steps, outputs)
            else:
                return total_loss / self.args.gradient_accumulation_steps
        
        return (total_loss, outputs) if return_outputs else total_loss
        
