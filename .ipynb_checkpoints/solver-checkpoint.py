#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
import random
from copy import deepcopy
import torch.nn as nn
torch.manual_seed(0)



class Solver:

    def __init__(
        self,
        model,
        optim,
        logger,
        loss_fn,
        val_fn,
        evo_optim,
        train_one,
        train_two,
        val,
        epochs=100,
        evo_step=5,
        child_count=60,
        best_child_count=3,
        mode = 'evo_cross',
        debug = True,
        lr = 0.001,
        device=0
        ):
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.val_fn = val_fn
        self.logger = logger
        self.evo_optim = evo_optim
        self.train_one = train_one
        self.train_two = train_two
        self.val = val
        self.epochs = epochs
        self.evo_step = evo_step
        self.child_count = child_count
        self.best_child_count = best_child_count
        self.mode = mode
        self.iteration = 0
        self.debug = debug
        self.lr = lr
        self.device = device
        
    def mutate_weights(self,l):
        # RANDOM GRADIENT UPDATE
        if 'Conv' in l.__class__.__name__:
            if l.weight.requires_grad:
                grad = l.weight.grad
                if not grad is None:
                    score = self.acc_score/50 # 50 wast foung as hyperparametr
                    temp = (self.lr)**(score)
                    #grad[grad <0] = 0
                    # CHANGE TO 0 - 1 to check later
                    uniform = torch.distributions.Uniform(-1,1)
                    noise = uniform.sample(sample_shape=l.weight.grad.size()).cuda()
                    #random_lr = self.lr*random.uniform(1, 5)
                    l.weight.data = l.weight.data - temp*noise*grad

    # The main call to start training
    
    def start(self):
        print ('Start training')
        self.model.eval()
        val_score = self.val_fn(self.model, self.val)
        print(f"started score - {val_score}")
        # START FROM FIRST EPOCH
        # NOW FIRST EPOCH WILL START WITH THE GRADIENT
        for epoch in range(1,self.epochs+1):
            if self.debug:
                print(f'Epoch: {epoch}\t Iterations: {self.iteration}')
            if (epoch % self.evo_step == 0) and (self.mode != 'gradient'):
                #self.model.eval()
                if self.mode == 'evo_only':
                    best_child_score = self.batch_evolve_simple()
                    self.logger.add_scalars({'Evolution accuracy':{'x':self.iteration,'y':best_child_score}})
                    if self.debug:
                        print(f"best child - {best_child_score}")
            else:
                self.model.train()
                (loss, val_score, train_score) = self.batch_train()
                self.logger.add_scalars({'Validation':{'x':self.iteration,'y':val_score}})
                self.logger.add_scalars({'Train_one':{'x':self.iteration,'y':train_score}})
                if self.debug:
                    print('[%d] loss: %.3f validation score: %.2f %%' \
                    % (epoch + 1, loss, val_score))
            self.model.eval()
            final_score = self.batch_test()
            self.logger.add_scalars({'Final score':{'x':self.iteration,'y':final_score}})
            self.logger.close()
        if self.debug:
            print ('Training is finished\nvalidation score: %.2f %%' \
            % final_score)
        self.logger.close()
        return self.model

    # Standard training
    def batch_train(self):
        loss = 0.0
        for (i, data) in tqdm(enumerate(self.train_one, 0)):
            (inputs, labels) = data
            inputs = inputs.cuda(self.device)
            labels = labels.cuda(self.device)

            self.optim.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optim.step()
            self.iteration+=1
            self.logger.add_scalars({'Training loss (only backprop)':{'x':self.iteration,'y':loss.item()}})
        val_score = self.val_fn(self.model, self.val)
        train_score = self.val_fn(self.model, self.train_one)
        
        # LAST BATCH LOSS 
        self.last_inputs = inputs
        self.last_labels = labels
        return (loss.item(), val_score, train_score)
    

    def batch_evolve_simple(self):
        
        best_child = deepcopy(self.model)   
        best_child_score,  bc_loss_score = self.val_fn(best_child, self.val, self.loss_fn)
        self.acc_score = best_child_score
        print(f'BASE SCORE VAL: acc: {best_child_score}, loss: {bc_loss_score}')
        best_child_score,  bc_loss_score = self.val_fn(best_child, self.train_two, self.loss_fn)
        print(f'BASE SCORE TRAIN_TWO: acc: {best_child_score}, loss: {bc_loss_score}')
        child_score = best_child_score
        for _ in range(self.child_count - 1):
            self.optim.zero_grad()
            data = next(iter(self.train_two))
            (inputs, labels) = data
            inputs = inputs.cuda(self.device)
            labels = labels.cuda(self.device)
                 
            child = deepcopy(self.model)
            
            self.iteration+=1
            self.model.train()
            
            outputs = child(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            child.apply(self.mutate_weights)
            
            #sort best score by train / val
            child_score, loss_score  = self.val_fn(child, self.train_two, self.loss_fn)           
            if self.debug:
                print('TRAIN_TWO: ch_acc_score',child_score, 'ch_loss', loss_score, 'best_score:',best_child_score)
            if child_score > best_child_score:
                bc_loss_score = loss_score
                best_child_score = child_score
                best_child = deepcopy(child)
                self.model = deepcopy(child)
      
        print('BEST TRAIN_TWO: ch_accuracy_score', best_child_score, 'ch_loss', bc_loss_score)
        self.logger.add_scalars({'Train_two':{'x':self.iteration,'y':best_child_score}})
        
        best_child_score,  bc_loss_score = self.val_fn(best_child, self.val, self.loss_fn)
        print('BEST VAL: ch_accuracy_score', best_child_score, 'ch_loss', bc_loss_score)
        self.logger.add_scalars({'Validation':{'x':self.iteration,'y':best_child_score}})
        
#         self.optim.param_groups = []
#         param_groups = list(self.model.parameters())
#         if not isinstance(param_groups[0], dict):
#             param_groups = [{'params': param_groups}]
#         for param_group in param_groups:
#             self.optim.add_param_group(param_group)
        del child
        del best_child
        return bc_loss_score
    
    def batch_test(self):
        return self.val_fn(self.model, self.val)
