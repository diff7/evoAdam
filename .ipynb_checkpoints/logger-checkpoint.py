import random
import os
import torch
import matplotlib
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

class Logger(SummaryWriter):
    def __init__(self, logdir):
        if not os.path.exists(logdir):
            os.mkdir(logdir)
            print(f"{logdir} was created")
        super(Logger, self).__init__(logdir)


    def add_scalars(self, values):
        '''values - dict where key is name of chart and value is a dict with iteration\'s number (x key) and y key is it\s value'''
        for k, v in values.items():
            self.add_scalar(k, v['y'], v['x'])


    def compare_models(self, names):
        for k, v in names.items():
            labels = v['model_name']
            scores = v['score']
            plt.bar(labels, scores, color='green')
            plt.xlabel("Model's name")
            plt.ylabel("It's score")
            plt.title(k)
            self.add_figure(k, plt.gcf(), global_step=v['iteration'])

if __name__ == "main":
    logger = Logger("./logs")
    logger.add_scalars({'Training loss (only backprop)':{'x':0,'y':0.9}, 'Validation loss (only backprop)':{'x':0,'y':0.99}})
    logger.compare_models({'Crossover test':{'iteration':1000, 'model_name':['A, B, C, D'], 'score':[27, 35, 14, 38]}})
    
