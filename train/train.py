import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from dataloader.dataloader import MaliciousCommentDataset
from model.mcc_classifier import MCCClassifier
from train.trainer import train_model

import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup, AdamW

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--train_fn', required=True, default='../data/malicious_comment_dataset.csv')
    p.add_argument('--label_columns', type=int, default=5)
    p.add_argument('--gpu_id', type=int, default=-1)

    config = p.parse_args()

    return config



def main(config):

    data = pd.read_csv(config.train_fn)
    temp_df, test_df = train_test_split(data, test_size=0.2, random_state = 216)
    train_df , valid_df = train_test_split(temp_df, test_size=0.2, random_state = 216)

    data_train = MaliciousCommentDataset(train_df, 512, True, False, label_cols=config.label_columns)
    data_test = MaliciousCommentDataset(test_df, 512, True, False, label_cols=config.label_columns)
    data_valid = MaliciousCommentDataset(valid_df, 512, True, False, label_cols=config.label_columns)

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=5, num_workers=3)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=5, num_workers=3)
    valid_dataloader = torch.utils.data.DataLoader(data_valid, batch_size=5, num_workers=3)


    model = MCCClassifier(num_classes=5, dr_rate = 0.45)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay' : 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr= 5e-6)
    crit=nn.BCEWithLogitsLoss()

    if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

    t_total = len(train_dataloader) * 200
    warmup_step = int(t_total * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    model, train_loss, valid_loss = train_model(train_dataloader,
                                                valid_dataloader, 
                                                model, 
                                                optimizer, 
                                                crit, 
                                                scheduler, 
                                                patience=10, 
                                                n_epochs=15, 
                                                path=f"../checkpoint/saved.pt" )


if __name__ == '__main__':
    config = define_argparser()
    main(config)