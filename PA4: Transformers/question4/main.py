import os, sys, pdb
import numpy as np
import random
import torch
from torch.optim import AdamW
import math

from tqdm import tqdm as progress_bar
from plot import plot_accuracy


from utils import set_seed, setup_gpus, check_directories
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
from model import ScenarioModel, SupConModel, CustomModel
from torch import nn
from transformers import get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_optimizer_with_LLRD(model, base_lr=3.5e-6, decay_rate=0.9):
    # set up optimizer with LLRD
    opt_parameters = []
    named_parameters = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    lr = base_lr  # Start with base learning rate

    # higher learning rate
    head_params = [p for n, p in named_parameters if "pooler" in n or "classifier" in n]
    opt_parameters.append({"params": head_params, "lr": base_lr * 1.05, "weight_decay": 0.01})

    # encoder layers
    for layer in range(11, -1, -1):  # Iterate from last (top) layer to first (bottom) layer
        layer_params = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n]
        opt_parameters.append({"params": layer_params, "lr": lr, "weight_decay": 0.01})
        lr *= decay_rate  # Apply decay for next layer

    # lowest learning rate
    embed_params = [p for n, p in named_parameters if "embeddings" in n]
    opt_parameters.append({"params": embed_params, "lr": lr, "weight_decay": 0.01})
    return AdamW(opt_parameters, lr=base_lr)


def baseline_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    total_steps = len(train_dataloader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.optimizer = optimizer
    model.scheduler = scheduler

    # store accuracy for plotting
    training_acc = []
    validation_acc = []
    
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()
        correct_train_preds = 0
        total_train_samples = 0

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            model.optimizer.step()
            model.scheduler.step()
            model.zero_grad()
            losses += loss.item()

            # Track training accuracy
            correct_train_preds += (logits.argmax(1) == labels).float().sum().item()
            total_train_samples += labels.size(0)

        train_accuracy = correct_train_preds / total_train_samples
        training_acc.append(train_accuracy)

        val_accuracy, _ = run_eval(args, model, datasets, tokenizer, split='validation')
        validation_acc.append(val_accuracy)

        print(f'Epoch {epoch_count+1} | Training Loss: {losses:.4f} | Training Accuracy: {train_accuracy:.4f} | Validation Accuracy: {val_accuracy:.4f}')

    final_train_accuracy, final_train_loss = run_eval(args, model, datasets, tokenizer, split='train')
    final_val_accuracy, final_val_loss = run_eval(args, model, datasets, tokenizer, split='validation')
    final_test_accuracy, final_test_loss = run_eval(args, model, datasets, tokenizer, split='test')

    # plot training vs validation accuracy
    plot_accuracy(training_acc, validation_acc, save_path="plots/baseline_training_vs_validation_accuracy.png")

    # save results
    with open("results.txt", "a") as f:
        f.write(f"Final Validation Accuracy: {final_val_accuracy:.4f} | Final Validation Loss: {final_val_loss:.4f}\n")
        f.write(f"Final Training Accuracy: {final_train_accuracy:.4f} | Final Training Loss: {final_train_loss:.4f}\n")
        f.write(f"Final Test Accuracy: {final_test_accuracy:.4f} | Final Test Loss: {final_test_loss:.4f}\n")

  
def custom_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    if args.use_llrd:
        optimizer = setup_optimizer_with_LLRD(model, base_lr=args.learning_rate, decay_rate=args.decay_rate)
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    total_steps = len(train_dataloader) * args.n_epochs
    if args.use_warmup:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_steps * total_steps),
            num_training_steps=total_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.optimizer = optimizer
    model.scheduler = scheduler

    # track accuracy for plotting
    training_acc = []
    validation_acc = []

    for epoch_count in range(args.n_epochs):
        model.train()
        losses = 0
        correct_train_preds = 0
        total_train_samples = 0

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            model.optimizer.step()
            model.scheduler.step()
            model.zero_grad()
            losses += loss.item()

            # Track training accuracy
            correct_train_preds += (logits.argmax(1) == labels).float().sum().item()
            total_train_samples += labels.size(0)

        train_accuracy = correct_train_preds / total_train_samples
        training_acc.append(train_accuracy)

        val_accuracy, _ = run_eval(args, model, datasets, tokenizer, split='validation')
        validation_acc.append(val_accuracy)

        print(f'Epoch {epoch_count+1} | Training Loss: {losses:.4f} | Training Accuracy: {train_accuracy:.4f} | Validation Accuracy: {val_accuracy:.4f}')

    final_train_accuracy, final_train_loss = run_eval(args, model, datasets, tokenizer, split='train')
    final_val_accuracy, final_val_loss = run_eval(args, model, datasets, tokenizer, split='validation')
    final_test_accuracy, final_test_loss = run_eval(args, model, datasets, tokenizer, split='test')

    # plot training vs validation accuracy
    plot_accuracy(training_acc, validation_acc, save_path="plots/custom_training_vs_validation_accuracy.png")

    # save results
    with open("results.txt", "a") as f:
        f.write(f"Final Validation Accuracy: {final_val_accuracy:.4f} | Final Validation Loss: {final_val_loss:.4f}\n")
        f.write(f"Final Training Accuracy: {final_train_accuracy:.4f} | Final Training Loss: {final_train_loss:.4f}\n")
        f.write(f"Final Test Accuracy: {final_test_accuracy:.4f} | Final Test Loss: {final_test_loss:.4f}\n")

        
def run_eval(args, model, datasets, tokenizer, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)

    criterion = nn.CrossEntropyLoss()  # Use the same loss function as training
    total_loss = 0  # Initialize loss tracking
    num_batches = 0
    acc = 0

    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch)
        logits = model(inputs, labels)

        # Compute loss
        loss = criterion(logits, labels)
        total_loss += loss.item()
        num_batches += 1

        # Compute accuracy
        correct_preds = (logits.argmax(1) == labels).float().sum().item()
        acc += correct_preds

    # Compute average loss and accuracy
    accuracy = acc / len(datasets[split])
    avg_loss = total_loss / num_batches  # Compute average test loss

    # Print results
    print(f'{split} Accuracy: {accuracy:.4f} | {split} Loss: {avg_loss:.4f} | Dataset Size: {len(datasets[split])}')

    # Log results
    with open("results.txt", "a") as f:
        f.write(f"{split} Accuracy: {accuracy:.4f} | {split} Loss: {avg_loss:.4f}\n")

    return accuracy, avg_loss  # Return both accuracy and loss



def supcon_train(args, model, datasets, tokenizer):
    from loss import SupConLoss
    criterion = SupConLoss()

    # task1: load training split of the dataset
    
    # task2: setup optimizer_scheduler in your model

    # task3: write a training loop for SupConLoss function 

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
    baseline_train(args, model, datasets, tokenizer)
  elif args.task == 'custom': # you can have multiple custom task for different techniques
    model = CustomModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    custom_train(args, model, datasets, tokenizer)
  elif args.task == 'supcon':
    model = SupConModel(args, tokenizer, target_size=60).to(device)
    supcon_train(args, model, datasets, tokenizer)
   
