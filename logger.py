import random
import os
import torch
import matplotlib
from datetime import datetime
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

class Logger(SummaryWriter):
    def __init__(self, logdir, experiment_note, folder_name='experiments'):
        today = datetime.now().strftime("%Y-%m-%d")
        time = datetime.now().strftime("%H:%M")
        
        folder_n = folder_name
        
        self.path = os.path.join(folder_n,today, logdir+'_'+time)
        
        if not os.path.exists('./'+today):
            os.makedirs(self.path, exist_ok=True)
        
        
        with open(os.path.join(self.path, 'experiment_info.txt'), 'w') as f:
            f.write(experiment_note+'\n')
        super(Logger, self).__init__(self.path)


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
            
    def add_post_result(self, note):
        with open(os.path.join(self.path, 'experiment_info.txt'),'a') as f:
            f.write(note+'\n')

if __name__ == "main":
    logger = Logger("./logs",'test')
    logger.add_scalars({'Training loss (only backprop)':{'x':0,'y':0.9}, 'Validation loss (only backprop)':{'x':0,'y':0.99}})
    logger.compare_models({'Crossover test':{'iteration':1000, 'model_name':['A, B, C, D'], 'score':[27, 35, 14, 38]}})
    
