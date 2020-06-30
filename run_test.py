import os
import sys
import copy
import random
import numpy as np
from datetime import datetime

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.optimizer import Optimizer
from logger import Logger

from solver import Solver
#from crossngover import CrossN

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# TRAIN_FULL
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=False, num_workers=2)

# TEST_FULL
valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

valloader = torch.utils.data.DataLoader(valset, batch_size=256,
                                         shuffle=False, num_workers=2)

# TRAIN_SPLITTED
# train_one, train_two = torch.utils.data.random_split(trainset, [45000,5000])

# train_one_loader = torch.utils.data.DataLoader(train_one, batch_size=256,
#                                           shuffle=False, num_workers=2)

# train_two_loader = torch.utils.data.DataLoader(train_two, batch_size=256,
#                                           shuffle=False, num_workers=2)




def train_models(params, net, device=0):

    mode = params['mode']
    experiment_note = ''
    path = ''
    for key in params:
        experiment_note += key +'_'+ str(params[key])+'\n'
        path +=  f'_{key}_'+str(params[key])+'_'
    logger = Logger(path, experiment_note)

    print(logger.path)
    print(experiment_note)

    #optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.SGD(net.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()
    #evo_optim = CrossN()
    evo_optim = None # from old crossingover experiments

    def validation(net, dataloader, loss_fn=None, device=0):
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images = images.cuda(device)
                labels = labels.cuda(device)
                outputs = net(images)
                if loss_fn:
                    total_loss += loss_fn(outputs, labels)
                    if loss_fn(outputs, labels) < 0:
                        ptint(loss_fn(outputs, labels))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if loss_fn is None: 
            return 100.0 * correct / total
        else:
            return 100.0 * correct / total, total_loss / total
    
    solver = Solver(
        net,
        optimizer,
        logger,
        criterion,
        validation,
        evo_optim,
        trainloader,
        valloader,
        epochs=50,
        evo_step=int(params['evo_step']),
        child_count=params['child_count'],
        best_child_count=3, # for crossingover, not used 
        mode=mode,
        debug=True,
        lr=params['lr'],
        device=device)

    logger.add_post_result(f'start: {datetime.now()}')
    solver.start()
    logger.add_post_result(f'finish: {datetime.now()}')
    torch.save(net.state_dict(), logger.path + '/model_last.chk')
    logger.close()


def init_model(pretrained=False):
    model = torchvision.models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.classifier = nn.Linear(num_ftrs, 10)
    model.cuda()
    return model
    

def train_main(model, params=None, to_stdout=False):
    
    if params is None:
        params = {'net_name':'Res18',
                 'preptrained':TF,
                 'child_count':40,
                 'evo_step':'5',
                 'random_seed':10}
        
    set_seed(params['random_seed'])
        
    modes = [ 'evo_only', 'gradient']  #'evo_cross'
    for mode in modes:
        params['mode'] = mode
        if to_stdout:
            orig_stdout = sys.stdout
            f = open(f'outputs/{name}_{mode}.txt', 'w')
            sys.stdout = f
    
        print(f'TRAINING MODE {mode.upper()}')
        
        # we copy the model to make sure both models
        # with sgd & noise were initialized equally 
        temp_model = copy.deepcopy(model)
        train_models(params, temp_model)
        
        if to_stdout:
            f.close()
            sys.stdout = orig_stdout
        torch.cuda.empty_cache()
    print('Finished')

if __name__ == "__main__":
    model = init_model(False)
    print('training')
    train_main(model, 'F', 'resnet18')
