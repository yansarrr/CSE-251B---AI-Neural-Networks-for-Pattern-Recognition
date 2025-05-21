import os, sys, pdb
import numpy as np
import random
import torch
from torch.optim import AdamW
import math
from utils_plot import plot_accuracy

from tqdm import tqdm as progress_bar

from utils import set_seed, setup_gpus, check_directories
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
from model import ScenarioModel, SupConModel, CustomModel
from torch import nn
from transformers import get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def baseline_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    # task2: setup model's optimizer_scheduler if you have
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    total_steps = len(train_dataloader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    model.optimizer = optimizer
    model.scheduler = scheduler
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            model.optimizer.step()  # backprop to update the weights
            model.scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()
    
        run_eval(args, model, datasets, tokenizer, split='validation')
        print('epoch', epoch_count, '| losses:', losses)
  
def custom_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader

    # task2: setup model's optimizer_scheduler if you have
      
    # task3: write a training loop

def run_eval(args, model, datasets, tokenizer, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)

    acc = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch)
        logits = model(inputs, labels) 
        pred = logits.argmax(dim=1)
        acc += (pred == labels).float().sum().item()
    
    accuracy = acc / len(datasets[split])
    print(f'{split} acc:', accuracy,
          f'| dataset split {split} size:', len(datasets[split]))
    return accuracy

def supcon_train(args, model, datasets, tokenizer, use_simclr=False):
    from loss import SupConLoss
    criterion = SupConLoss()

    train_dataloader = get_dataloader(args, datasets['train'], 'train')
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    total_steps = len(train_dataloader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    model.optimizer = optimizer
    model.scheduler = scheduler

    model.args.contrast_mode = True

    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch)

            features = model(inputs, labels).unsqueeze(1)
            loss = criterion(features, labels=None if use_simclr else labels)
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses += loss.item()

        print(f'[SupCon] Epoch {epoch_count} | Loss: {losses:.4f}')
        run_eval(args, model, datasets, tokenizer, split='validation')
    

def classification_finetune(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()
    train_dataloader = get_dataloader(args, datasets['train'], 'train')

    optimizer = AdamW(model.parameters(), lr=5e-5, eps=args.adam_epsilon)
    total_steps = len(train_dataloader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    model.args.contrast_mode = False

    train_acc_list = []
    val_acc_list = []

    for epoch_count in range(args.n_epochs):
        model.train()
        total_loss = 0.0
        
        correct = 0
        total_samples = 0

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).float().sum().item()
            total_samples += len(labels)

        train_acc = correct / total_samples
        train_acc_list.append(train_acc)

        val_acc = run_eval(args, model, datasets, tokenizer, split='validation')
        val_acc_list.append(val_acc)

        print(f'[ClsFinetune] Epoch {epoch_count} | Loss: {total_loss:.4f} | Train_Acc: {train_acc:.4f}')
    
    plot_accuracy(train_acc_list, val_acc_list,
                  save_path="supcon_or_simclr_cls_finetune_acc.png")

if __name__ == "__main__":
  args = params()
  args = setup_gpus(args)
  args = check_directories(args)
  set_seed(args)

  cache_results, already_exist = check_cache(args)
  tokenizer = load_tokenizer(args)

  if already_exist:
    features = cache_results
  else:
    data = load_data()
    features = prepare_features(args, data, tokenizer, cache_results)
  datasets = process_data(args, features, tokenizer)
  for k,v in datasets.items():
    print(k, len(v))
 
  if args.task == 'baseline':
    model = ScenarioModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    baseline_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')
  elif args.task == 'custom': 
    model = CustomModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    custom_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')
  elif args.task == 'supcon':
    model = SupConModel(args, tokenizer, target_size=60).to(device)

    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')

    supcon_train(args, model, datasets, tokenizer, use_simclr=False)
    classification_finetune(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')


  elif args.task == 'simclr':
    model = SupConModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')

    supcon_train(args, model, datasets, tokenizer, use_simclr=True)

    classification_finetune(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')
   
