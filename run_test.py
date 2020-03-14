import sys
import copy
from datetime import datetime

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.optimizer import Optimizer, required

from logger import Logger

from solver import Solver
from crossngover import CrossN

torch.manual_seed(0)



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# TRAIN_FULL
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader_full = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=False, num_workers=2)

# TEST_FULL
valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

valloader = torch.utils.data.DataLoader(valset, batch_size=256,
                                         shuffle=False, num_workers=2)

# TRAIN_SPLITTED
train_one, train_two = torch.utils.data.random_split(trainset, [48000,2000])

train_one_loader = torch.utils.data.DataLoader(train_one, batch_size=256,
                                          shuffle=False, num_workers=2)

train_two_loader = torch.utils.data.DataLoader(train_two, batch_size=256,
                                          shuffle=False, num_workers=2)




def train_models(params, net, device=0):

    mode = params['mode']
    evo_step = int(params['evo_step'])

    experiment_note = ''
    path = ''
    for key in params:
        experiment_note += key +'_'+ params[key]+'\n'
        path +=  '_'+params[key]+'_SGD'


    logger = Logger(path, experiment_note)

    print(logger.path)
    print(experiment_note)

    lr = 0.001

    #optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    evo_optim = CrossN()

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

    
    if mode == 'evo_only':
        trainloader = train_one_loader
        print('USING TRAIN ONE & TRAIN TWO')
    if mode == 'gradient':
        trainloader = trainloader_full
        print('USING FULL TRAIN SET')
    
    solver = Solver(
        net,
        optimizer,
        logger,
        criterion,
        validation,
        evo_optim,
        trainloader,
        train_two_loader,
        valloader,
        epochs=50,
        evo_step=evo_step,
        child_count=40,
        best_child_count=3,
        mode=mode,
        debug=True,
        lr=lr,
        device=device)

    logger.add_post_result(f'start: {datetime.now()}')
    solver.start()
    logger.add_post_result(f'finish: {datetime.now()}')
    torch.save(net.state_dict(), logger.path + '/model_last.chk')
    logger.close()



def train_three_types(model, TF, name):
    modes = [ 'evo_only', 'gradient']  #'evo_cross'

    evo_step = 5

    for mode in modes:
#         orig_stdout = sys.stdout
#         f = open(f'outputs/{name}_{mode}.txt', 'w')
#         sys.stdout = f
    
        print(f'TRAINING MODE {mode.upper()}')
        params = {'net_name':name,
             'preptrained':TF,
             'mode':mode,
             'evo_step':str(evo_step)}
        temp_model = copy.deepcopy(model)
        train_models(params, temp_model)
#         f.close()
#         sys.stdout = orig_stdout
        torch.cuda.empty_cache()
    print('Finished')

if __name__ == "__main__":
    net = torchvision.models.resnet18(pretrained=False)

    num_ftrs = net.fc.in_features

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net.classifier = nn.Linear(num_ftrs, len(classes))
    net.cuda()
    print('training')
    train_three_types(net, 'F', 'resnet18')
