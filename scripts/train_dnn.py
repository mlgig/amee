import os
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle as pkl
from utils.data import LocalDataLoader
from typing import cast, Any, Dict, List, Tuple, Optional


device = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer():
    def __init__(self,model,datapath,ds):
        self.model = model
        self.datapath = datapath
        self.ds = ds
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []
        self.test_result = {}
    
    def fit(self,batch_size=64,num_epochs=100,val_size=0.2,
        learning_rate=0.01,patience=20,save_best_model=True):
        data = LocalDataLoader(self.datapath,self.ds)
        train_loader,val_loader = data.get_loaders(mode='train')
        
        optimizer=torch.optim.Adam(self.model.parameters(),lr=learning_rate)
        
        best_val_loss = np.inf
        best_val_acc = 0
        patience_counter = 0
        best_state_dic = None

        self.model.train()
        for epoch in range(num_epochs):
            epoch_train_loss = []
            for X_train,y_train in train_loader:
                X_train,y_train = X_train.to(device),y_train.to(device)
                optimizer.zero_grad()
                output = self.model(X_train)

                if y_train.shape[-1]==2:
                    train_loss = F.binary_cross_entropy_with_logits(
                        output, y_train.float(), reduction='mean'
                        )
                    
                else:
                    train_loss = F.cross_entropy(
                        output,y_train.argmax(dim=-1), reduction='mean')
                epoch_train_loss.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
            self.train_loss.append(np.mean(epoch_train_loss))

            epoch_val_loss = []
            self.model.eval()
            true_list = []
            pred_list = []
            for X_val,y_val in val_loader:
                with torch.no_grad():
                    output = self.model(X_val)
                    if y_val.shape[-1]==2:
                        val_loss = F.binary_cross_entropy_with_logits(
                            output, y_val.float(), reduction='mean'
                            ).item()
                    else:
                        val_loss = F.cross_entropy(output,
                                    y_val.argmax(dim=-1), reduction='mean').item()
                    epoch_val_loss.append(val_loss)

                    true_list.append(y_val.detach().numpy())
                    preds = self.model(X_val)
                    preds=torch.softmax(preds,dim=-1)
                    pred_list.append(preds.detach().numpy())
            true_np,preds_np = np.concatenate(true_list), np.concatenate(pred_list)
            true_np = np.argmax(true_np,axis=-1)
            preds_np= np.argmax(preds_np,axis=-1)
            val_acc = accuracy_score(true_np,preds_np)
            self.val_acc.append(val_acc)
            self.val_loss.append(np.mean(epoch_val_loss))

            print(f'Epoch: {epoch + 1}, '
              f'Train loss: {round(self.train_loss[-1], 4)}, '
              f'Val loss: {round(self.val_loss[-1], 4)}, ',
              f'Val acc: {round(val_acc, 4)} ')

            if self.val_loss[-1] < best_val_loss:
                best_val_loss = self.val_loss[-1]
            # if self.val_acc[-1] > best_val_acc:
            #     best_val_acc = self.val_acc[-1]
                best_state_dict = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

                if patience_counter == patience:
                    if best_state_dict is not None:
                        self.model.load_state_dict(cast(Dict[str, torch.Tensor], best_state_dict))
                    print('Early stopping!')
                    return
                    
        if save_best_model == True:
            savepath = './model/{self.ds}_best.pkl'
            torch.save(self.model,savepath)

    def evaluate(self,):
        data = LocalDataLoader(self.datapath,self.ds)
        test_loader,_ = data.get_loaders(mode='test')
        self.model.eval()
        true_list,pred_list = [],[]

        for x,y in test_loader:
            with torch.no_grad():
                true_list.append(y.detach().numpy())
                preds = self.model(x)
                if y.shape[-1] == 2:
                    preds = torch.sigmoid(preds)
                else:
                    preds=torch.softmax(preds,dim=-1)
                pred_list.append(preds.detach().numpy())
        true_np,preds_np = np.concatenate(true_list), np.concatenate(pred_list)

        true_np = np.argmax(true_np,axis=-1)
        preds_np= np.argmax(preds_np,axis=-1)
        self.test_result = accuracy_score(true_np,preds_np)

        print(f'Accuracy score: {round(self.test_result, 4)}')

