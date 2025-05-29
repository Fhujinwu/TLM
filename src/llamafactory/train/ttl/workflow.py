from typing import TYPE_CHECKING, List, Optional


from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import cal_effective_tokens, get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

import torch
import torch.nn as nn

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ...hparams import FinetuningArguments, ModelArguments

logger = logging.get_logger(__name__)

class TTLModel(nn.Module):
    def __init__(self, 
                 data_args: "DataArguments",
                 model_args: "ModelArguments", 
                 training_args: "Seq2SeqTrainingArguments", 
                 finetuning_args: "FinetuningArguments", 
                 generating_args: "GeneratingArguments",
                tokenizer_module,
                template,
                model
                ):
        super().__init__()
        self.data_args = data_args
        self.training_args = training_args
        self.finetuning_args = finetuning_args
        self.model_args = model_args
        self.generating_args = generating_args
        self.template = template

        self.tokenizer_module = tokenizer_module
        self.tokenizer = self.tokenizer_module["tokenizer"]
        
        self.model = model
        
        self.base_output_dir = self.training_args.output_dir
        
    
    def reset_trainer(self, train_dataset, **kwargs):
        data_collator = SFTDataCollatorWith4DAttentionMask(
            template=self.template,
            # pad_to_multiple_of=8 if self.training_args.do_train else None,  # for shift short attention
            pad_to_multiple_of= None,  # for shift short attention
            label_pad_token_id=IGNORE_INDEX if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id,
            block_diag_attn=self.model_args.block_diag_attn,
            attn_implementation=getattr(self.model.config, "_attn_implementation", None),
            compute_dtype=self.model_args.compute_dtype,
            **self.tokenizer_module,
        )

        self.trainer = CustomSeq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            finetuning_args=self.finetuning_args,
            model_args=self.model_args,
            data_args=self.data_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            **self.tokenizer_module,
            **kwargs
        )
    
    def forward(self, train_batch, predict_batch):
        if self.finetuning_args.setting == "offline_ttl":
            self.forward_for_offline(train_batch=train_batch, predict_batch=predict_batch)
        
        elif self.finetuning_args.setting == "online_ttl":
            self.forward_for_online(train_batch=train_batch, predict_batch=predict_batch)

        else:
            raise ValueError(
                f'NO such setting: {self.finetuning_args.setting}'
            )

    def forward_for_offline(self, train_batch, predict_batch):  
        """
        First Train, then Predict.
        This is the offline TTL setting, where we first train the model using only the inputs, then use the trained model to predict the results of the training data.
        """
        # train
        self.tokenizer.padding_side = "right"  # use right-padding in training
        self.training_args.generation_max_length = self.training_args.generation_max_length or self.data_args.cutoff_len
        self.training_args.generation_num_beams = self.data_args.eval_num_beams or self.training_args.generation_num_beams
        self.training_args.remove_unused_columns = False  # important for multimodal dataset
        self.reset_trainer(train_dataset=train_batch)
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model()

        self.unwrap_model()

        # predict
        gen_kwargs = self.generating_args.to_dict()
        gen_kwargs["eos_token_id"] = [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids
        gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        gen_kwargs["logits_processor"] = get_logits_processor()
        # decoder-only models must use left-padding for batched generation.
        if self.training_args.predict_with_generate:
            self.tokenizer.padding_side = "left"  # use left-padding in generation
        self.training_args.output_dir = self.base_output_dir + f'/predict-temperature_{self.generating_args.temperature}-max_new_tokens_{self.generating_args.max_new_tokens}'
        
        self.reset_trainer(train_dataset=None)
        predict_results = self.trainer.predict(predict_batch, metric_key_prefix="predict", **gen_kwargs)
        self.trainer.save_predictions(predict_batch, predict_results)
    

    def forward_for_online(self, train_batch, predict_batch):
        """
        First Predict, then Train.
        This is the online TTL setting, where we first predict the results of the training data, then train the model with only the inputs.
        """
        ####################################
        # use the latest model to predict
        ####################################

        self.training_args.output_dir = self.base_output_dir + f'/predict-temperature_{self.generating_args.temperature}-max_new_tokens_{self.generating_args.max_new_tokens}'  # the folder to save prediction results
        # Keyword arguments for `model.generate`
        gen_kwargs = self.generating_args.to_dict()
        gen_kwargs["eos_token_id"] = [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids
        gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        gen_kwargs["logits_processor"] = get_logits_processor()
        # decoder-only models must use left-padding for batched generation.
        if self.training_args.predict_with_generate:
            self.tokenizer.padding_side = "left"  # use left-padding in generation
        
        self.training_args.generation_max_length = self.training_args.generation_max_length or self.data_args.cutoff_len
        self.training_args.generation_num_beams = self.data_args.eval_num_beams or self.training_args.generation_num_beams
        self.training_args.remove_unused_columns = False  # important for multimodal dataset
        
        self.reset_trainer(train_dataset=None)
        predict_results = self.trainer.predict(predict_batch, metric_key_prefix="predict", **gen_kwargs)
        self.trainer.save_predictions(predict_batch, predict_results)
        
        self.unwrap_model()

        self.training_args.output_dir = self.base_output_dir  # 保存 adapter 的文件夹
        self.tokenizer.padding_side = "right"

        self.reset_trainer(train_dataset=train_batch)
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model()    # 保存模型到 training_args.output_dir

        self.unwrap_model()

    def unwrap_model(self):
        self.model = self.trainer.accelerator.unwrap_model(self.model, keep_fp32_wrapper=False)
    

    
def run_ttl(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="ttl", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    
    train_dataset = dataset_module['train_dataset']
    eval_dataset = dataset_module['eval_dataset']
    print(train_dataset, eval_dataset)
    print(train_dataset[0], '\n', eval_dataset[0])
    

    ttl_model = TTLModel(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        tokenizer_module=tokenizer_module,
        template=template, 
        model=model
    )
    
    if finetuning_args.setting == "offline_ttl":
        ttl_model.forward(train_batch=train_dataset, predict_batch=eval_dataset)
    elif finetuning_args.setting == "online_ttl":
        streaming_batch_size = finetuning_args.streaming_batch_size
        num_of_batch = len(train_dataset) // streaming_batch_size
        if len(train_dataset) % streaming_batch_size != 0:
            num_of_batch += 1
        for k in range(num_of_batch):
            logger.info_rank0(f"Processing batch {k+1}/{num_of_batch} with streaming batch size {streaming_batch_size}")
            if (k+1)*streaming_batch_size > len(train_dataset):
                end_index = len(train_dataset)
            else:
                end_index = (k+1)*streaming_batch_size
            sub_trainset = train_dataset.select(range(k*streaming_batch_size, end_index))
            sub_evalset = eval_dataset.select(range(k*streaming_batch_size, end_index))
            ttl_model.forward(train_batch=sub_trainset, predict_batch=sub_evalset)
    else:
        raise ValueError(
            f'NO such setting: {finetuning_args.setting}'
        )