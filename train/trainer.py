import torch
import numpy as np
from metrics import calculate_metrics

from model.mcc_classifier import EarlyStopping


def train_model(train_dataloader, valid_dataloader, model, optimizer, crit, scheduler, patience, n_epochs, path, device):
    
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = [] 

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)

    for epoch in range(1, n_epochs + 1):
        
        # initialize the early_stopping object
        model.train()
        train_epoch_pred=[]
        train_loss_record=[]

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
            optimizer.zero_grad()

            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
        
            label = label.float().to(device)

            out= model(token_ids, valid_length, segment_ids)#.squeeze(1)
            
            loss = crit(out, label)

            train_loss_record.append(loss)

            train_pred=out.detach().cpu().numpy()
            train_real=label.detach().cpu().numpy()

            train_batch_result = calculate_metrics(np.array(train_pred), np.array(train_real))
            
            if batch_id%50==0:
                print(f"batch number {batch_id}, train col-wise accuracy is : {train_batch_result['Column-wise Accuracy']}")
                
            # save prediction result for calculation of accuracy per batch
            train_epoch_pred.append(train_pred)
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule

            train_losses.append(loss.item())

        train_epoch_pred=np.concatenate(train_epoch_pred)
        train_epoch_target=train_dataloader.dataset.labels
        train_epoch_result=calculate_metrics(target=train_epoch_target, pred=train_epoch_pred)
        
        print(f"=====Training Report: mean loss is {sum(train_loss_record)/len(train_loss_record)}=====")
        print(train_epoch_result)
        
        print("=====train done!=====")


        # eval
        test_epoch_pred=[]
        test_loss_record=[]

        model.eval()
        with torch.no_grad():
            for batch_id, (token_ids, valid_length, segment_ids, test_label) in enumerate(valid_dataloader): # 여기 valid로 수정
                
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                valid_length = valid_length
                
                # test_label = test_label.long().to(device)
                test_label = test_label.float().to(device)

                test_out = model(token_ids, valid_length, segment_ids)

                test_loss = crit(test_out, test_label)

                test_loss_record.append(test_loss)
                
                valid_losses.append(test_loss.item())

                test_pred=test_out.detach().cpu().numpy()
                test_real=test_label.detach().cpu().numpy()

                test_batch_result = calculate_metrics(np.array(test_pred), np.array(test_real))

                if batch_id%50==0:
                    print(f"batch number {batch_id}, test col-wise accuracy is : {test_batch_result['Column-wise Accuracy']}")

                # save prediction result for calculation of accuracy per epoch
                test_epoch_pred.append(test_pred)

        test_epoch_pred=np.concatenate(test_epoch_pred)
        test_epoch_target=valid_dataloader.dataset.labels
        test_epoch_result=calculate_metrics(target=test_epoch_target, pred=test_epoch_pred)

        print(f"=====Testing Report: mean loss is {sum(test_loss_record)/len(test_loss_record)}=====")
        print(test_epoch_result)

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(path))

    return  model, avg_train_losses, avg_valid_losses