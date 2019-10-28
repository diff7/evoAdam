import itertools 
from copy import copy
import numpy as np



class CrossN:
    def __init__(self, params, n_combinations = 2, pmix = 0.5):
        self.p = pmix
        self.params = params
        self.n_combinations = n_combinations
        self.layers = []
        
    def make_ancestry(self, nets):
        n = []
        abc = ['A','B','C','D','E','F']
        for counter, net in enumerate(nets):
            if not 'ancestry' in net.__dir__():
                if len(abc) > counter:
                    net.ancestry = abc[counter]
                else: 
                    net.ancestry = str(counter)
            n.append(net)
        return n 
 
    def make_combinations(self, nets):
        nets = self.make_ancestry(nets)
        parents = [n.ancestry for n in nets]
        families = itertools.product(parents,repeat = 
                                           min(len(parents), 
                                              self.n_combinations))
        return families
    
    def find_layers(self, model, keyword = 'weight'):
        model_state_dict = model.state_dict()
        self.layers = [layer_name for layer_name in model_state_dict 
                       if keyword in layer_name]
        
    
    def switch_weights(self, parentA, parentB, layers):
        
        # abundant copying may lead to unnessary memmory usage - need to optimize
        parentA_state_dict = parentA.state_dict()
        parentB_state_dict = parentB.state_dict()
        
        if parentA.ancestry != parentB.ancestry:
            # p is responsible for number or False values to be replaced
            p = self.p
            child_params = copy(parentA_state_dict)
            for layer in self.layers:
                w = parentB_state_dict[layer]
                shape = w.shape
                #mask =  np.random.choice([False, True], size=(shape), p=[1-p, p])
                mask = torch.cuda.FloatTensor(shape).uniform_() > p
                #switching weights
                print(layer, w[mask].shape)
                print(layer, child_params[layer][mask].shape)
                try:
                    child_params[layer][mask]=w[mask]
                except Exception as e:
                    print('ERR:', e,layer)
            ancestry = '({}{}+{}{})'.format(str(p), parentA.ancestry ,str(1-p), parentB.ancestry)
            #HOWTO initiate a new model in any other way?
            child = PConvUNet()
            child.load_state_dict(child)
            child.ancestry = ancestry 
        else:
            child = parentA
            ancestry = '({}+{})'.format(parentA.ancestry, parentB.ancestry)
            child.ancestry = ancestry 
            
        return child
    
#     def mutate(self, ):
        
        
#     def sparta(self, ):
            
        