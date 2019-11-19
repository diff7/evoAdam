#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
import random
from copy import deepcopy

# To keep limited number of items in the list

class MaxSizeList(object):

    def __init__(self, max_length):
        self.max_length = max_length
        self.ls = []

    def push(self, st):
        if len(self.ls) == self.max_length:
            self.ls.pop(0)
        self.ls.append(st)

    def get_list(self):
        return self.ls


loc = 0
scale = 0.001
normal = torch.distributions.Normal(loc, scale)  # create a normal distribution object


# Mutate weights if a layer is the weights layer

def mutate_weights(model, keyword='weight'):
    model_state_dict = model.state_dict()
    for layer_name in model_state_dict:
        if keyword in layer_name:
            model_state_dict[layer_name] += \
                normal.rsample(model_state_dict[layer_name].size()) #.cuda()
    model.load_state_dict(model_state_dict)

    return model


class Solver:

    def __init__(
        self,
        model,
        optim,
        logger,
        loss_fn,
        val_fn,
        evo_optim,
        train,
        val,
        test,
        epochs=100,
        evo_step=1,
        child_count=20,
        best_child_count=3,
        mode = 'normal'
        ):
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.val_fn = val_fn
        self.logger = logger
        self.evo_optim = evo_optim
        self.train = train
        self.val = val
        self.test = test
        self.epochs = epochs
        self.evo_step = evo_step
        self.child_count = child_count
        self.best_child_count = best_child_count
        self.mode = mode
        self.iteration = 0
    
    # The main call to start training
    def start(self):
        print ('Start training')
        for epoch in range(self.epochs):
            print(f'Epoch: {epoch}')
            if epoch % self.evo_step == 0:
                self.model.eval()
                if self.mode == 'normal':
                    best_child_score = self.batch_evolve_normal()
                elif self.mode == 'simple':
                    best_child_score = self.batch_evolve_simple()
                self.logger.add_scalars({'Evolution accuracy':{'x':self.iteration,'y':best_child_score}}
                print(f"best child - {best_child_score}%")
            else:
                self.model.train()
                (loss, val_score) = self.batch_train()
                self.logger.add_scalars({'Validation':{'x':self.iteration,'y':val_score}}
                print ('[%d] loss: %.3f validation score: %.2f %%' \
                    % (epoch + 1, loss, val_score))
        self.model.eval()
        final_score = self.batch_test()
        print ('Training is finished\nvalidation score: %.2f %%' \
            % final_score)
        return self.model

    # Standard training
    def batch_train(self):
        loss = 0.0
        for (i, data) in enumerate(self.train, 0):
            (inputs, labels) = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            self.optim.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optim.step()
            self.iteration+=1
            self.logger.add_scalars({'Training loss (only backprop)':{'x':self.iteration,'y':loss.item()}}
        val_score = self.val_fn(self.model, self.val)
        return (loss.item(), val_score)
    

    # Mutate weights N times, choose 3 best candidates
    # Mix 3 best models with each ohther using cross_N
    def batch_evolve_normal(self):
        Logger = self.logger
        best_kids = MaxSizeList(self.best_child_count)
        best_child = deepcopy(self.model)
        best_child.apply(mutate_weights)
        best_child_score = self.val_fn(best_child, self.val)
        best_kids.push(best_child)
        for _ in range(self.child_count - 1):
            child = deepcopy(self.model)
            child.apply(mutate_weights)
            child_score = self.val_fn(child, self.val)
            if child_score > best_child_score:
                best_child_score = child_score
                best_child = deepcopy(child)
                best_kids.push(best_child)
        for child in self.evo_optim.breed(best_kids.get_list()):
            child_score = self.val_fn(child, self.val)
            if child_score > best_child_score:
                best_child_score = child_score
                best_child = deepcopy(child)
        self.model = deepcopy(best_child)
        self.optim.param_groups = []
        param_groups = list(self.model.parameters())
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        for param_group in param_groups:
            self.optim.add_param_group(param_group)
        del child
        del best_child
        return best_child_score
    
    # Mutate weights N times, choose 3 best candidates
    def batch_evolve_simple(self):
        best_child = deepcopy(self.model)
        best_child.apply(mutate_weights)
        best_child_score = self.val_fn(best_child, self.val)
        for _ in range(self.child_count - 1):
            child = deepcopy(self.model)
            child.apply(mutate_weights)
            child_score = self.val_fn(child, self.val)
            if child_score > best_child_score:
                best_child_score = child_score
                best_child = deepcopy(child)
                self.model = deepcopy(best_child)  
        self.optim.param_groups = []
        param_groups = list(self.model.parameters())
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        for param_group in param_groups:
            self.optim.add_param_group(param_group)
        del child
        del best_child
        return best_child_score
    
    def batch_test(self):
        return self.val_fn(self.model, self.test)
