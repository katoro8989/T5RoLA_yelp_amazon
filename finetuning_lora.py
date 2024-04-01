import os
import time
import torch
import numpy as np
from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration, T5Tokenizer
import dataset
import model
import argparse
import datetime
import wandb
import evaluate
import peft

def finetune(args):
    train_dataset = args.dataset
    
    run_id = f'rola_{args.model}'
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f'{run_id}_{timestamp}'
    output_dir = os.path.join(args.output_dir, run_id)

    assert train_dataset is not None, "Please provide a training dataset."
    dataset_class = dataset.get_dataset(args.dataset, args.model)
    
    if args.checkpoint_path == None:
        model_class = model.get_model(args.model)
    else:
        model_class = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path)
    
    model_class.enable_input_require_grads()
    model_class.gradient_checkpointing_enable()
    
    peft_comfig = peft.LoraConfig(
        task_type=peft.TaskType.SEQ_2_SEQ_LM, 
        r=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout, 
        inference_mode=False, 
        target_modules=["k","q","v","o"], 
    )
    
    model_class = peft.get_peft_model(model_class, peft_comfig)
    model_class.print_trainable_parameters()
    
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    
    if args.wandb:
        report = "wandb"
        print("=====wandb logging starts=====")
        wandb.init(project="t5-finrtunes",
            name=run_id,
            group="katoro13")
    else:
        report = None
    
    training_args = TrainingArguments(
        output_dir=output_dir,          
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        learning_rate=args.lr,        
        per_device_train_batch_size=args.train_batch_size,  
        per_device_eval_batch_size=args.eval_batch_size,   
        warmup_steps=args.warmup_steps,                
        weight_decay=args.weight_decay,               
        logging_strategy=args.logging_strategy, 
        run_name=run_id,  
        report_to=report, 
        fp16=args.fp16, 
        logging_dir='./logs', 
        evaluation_strategy=args.evaluation_strategy, 
        fp16_full_eval=args.fp16_full_eval, 
        eval_steps=args.eval_steps, 
        eval_accumulation_steps=args.eval_accumulation_steps, 
        auto_find_batch_size=args.auto_find_batch_size, 
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
    )
    
    print(training_args.device)
    
    sample_batch = next(iter(dataset_class["train"]))
    print(np.array(sample_batch["input_ids"]).shape)
    print(np.array(sample_batch["attention_mask"]).shape)
    print(np.array(sample_batch["labels"]).shape)
    
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        predictions = [pred if pred in ['0', '1'] else '2' for pred in predictions] 
        
        return metric.compute(predictions=predictions, references=labels)

    
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    
    trainer = Trainer(
        model=model_class,
        args=training_args,
        train_dataset=dataset_class["train"],
        eval_dataset=dataset_class["test"],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetuning of T5')
    parser.add_argument('--dataset', type=str, default="yelp_polarity")
    parser.add_argument('--model', type=str, default="t5-small")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--epochs', type=float, default=1.)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=int, default=0)
    parser.add_argument('--fp16', action='store_true', help='whether fp16')
    parser.add_argument('--logging_dir', type=int, default=None)
    parser.add_argument('--logging_strategy', type=str, default="epoch")
    parser.add_argument('--evaluation_strategy', type=str, default="epoch")
    parser.add_argument('--wandb', action='store_true', help='whether log on wandb')
    parser.add_argument('--exp_id', type=str, default=None, help='exp id for reporting')
    parser.add_argument('--fp16_full_eval', action='store_true')
    parser.add_argument('--eval_steps', type=float, default=1.)
    parser.add_argument('--eval_accumulation_steps', type=int)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--auto_find_batch_size', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=8)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--ddp_find_unused_parameters', action='store_true')
    args = parser.parse_args()
                    
    
    finetune(args)
