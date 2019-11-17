import itertools 
from copy import deepcopy, copy
import numpy as np
from tqdm import tqdm
import torch



class CrossN:
    def __init__(self, n_combinations = 2, pmix = 0.5, mode = 'mask'):
        self.p = pmix
        self.n_combinations = n_combinations
        self.layers = []
        self.mode = mode
    
    # Create an ancestry object for model if does not exist
    def make_ancestry(self, nets):
        n = []
        abc = ['A','B','C','D','E','F']
        for counter, net in enumerate(nets):
            if not 'ancestry' in net.__dir__():
                if len(abc) > counter:
                    net.ancestry = '[{}]'.format(abc[counter])
                else: 
                    net.ancestry = str(counter)
            n.append(net)
        return n 
 
    # Make combinations for parent - parent pairs
    def make_combinations(self, nets):
        nets = self.make_ancestry(nets)
        parents = [n.ancestry for n in nets]
        pairs = itertools.product(parents,repeat = 
                                           min(len(parents), 
                                              self.n_combinations))

        return pairs
    
    # Find all layers names to mix 
    def find_layers(self, model, keyword = 'weight'):
        model_state_dict = model.state_dict()
        self.layers = [layer_name for layer_name in model_state_dict 
                       if keyword in layer_name]
        
    
    # Mix weights or use additive mixing child += p*parent
    def switch_weights(self, parentA, parentB):
        
        # abundant copying may lead to unnessary memmory usage - need to optimize
        child = deepcopy(parentA) 
        parent_B = deepcopy(parentB)
        parentB_state_dict = parent_B.state_dict()
        # p is responsible for percantage or False / Zero values, values not to be replaced
        p = self.p
        
        if parentA.ancestry != parent_B.ancestry:
            
            child_params = child.state_dict()
            
            if self.mode == 'mask':
              for layer in self.layers:
                  w = parentB_state_dict[layer]
                  shape = w.shape
                  
                  # creating the mask with tensor does not cause shape mismatch problem 
                  #mask =  np.random.choice([False, True], size=(shape), p=[1-p, p])
                  mask = torch.cuda.FloatTensor(shape).uniform_() > p
                  # switching weights
                  try:
                      child_params[layer][mask]=w[mask]
                      #child_params[layer][mask]=child_params[layer][mask]=w[mask].view(shape)
                  except Exception as e:
                      print('skipping layer: ',layer, e)
              #HOWTO initiate a new model in any other way?
            
            elif self.mode == 'addition': 
              for layer in self.layers:
                  w = parentB_state_dict[layer]
                  try:
                      child_params[layer]+=p*w
                  except Exception as e:
                      print('skipping layer: ',layer, e)

            
            child.load_state_dict(child_params)
          
        child.ancestry +=',{}.{}'.format(round(1-p, 2), parent_B.ancestry.replace(',',' '))
        return child
    
    # Basicaly the main function to call which combines all the others
    def breed(self, nets: list()):
        nets = deepcopy(nets)
        self.find_layers(nets[0])
        families = self.make_combinations(nets)
        children = []
        nets_named = {n.ancestry:n for n in nets}
        for family in families:
            parents = [nets_named[f] for f in family]
            child = self.switch_weights(*parents)
            yield child
    
    def history(self, net):
        for i, p in enumerate(net.ancestry.split(',')):
            print('{}+{}'.format('  '*i,p))

