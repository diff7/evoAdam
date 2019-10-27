import itertools 
from copy import copy



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
        families = itertools.combinations_with_replacement(parents,
                                           min(len(parents), 
                                               self.n_combinations))
        return families
    
    def find_layers(model_state_dict, keyword = 'weight'):
        self.layers = [layer_name for layer_name in model_state_dict 
                       if keyword in layer_name]
        
    
    def switch_weights(self, parentA, parentB):
        
        # abundant copying may lead to unnessary memmory usage - need to optimize
        parentA_state_dict = parentA.state_dict()
        parentB_state_dict = parentB.state_dict()
        
        if parentA.ancestry != parentB.ancestry:
            # p is responsible for number or True values to be replaced
            p = self.p
            child_params = copy(parentA_state_dict)
            for layer in self.layers:
                w = parentB_state_dict[layer]
                mask =  np.random.choice([False, True], size=(w.shape), p=[1-p, p])
                #switching weights
                child_params[layer][mask]=w[mask]
            ancestry = '({}{}+{}{})'.format(str(p), parentA.ancestry ,str(1-p), parentB.ancestry)
            #HOWTO initiate a new model in any other way?
            child = parentA
            child.ancestry = ancestry 
            child.loaload_state_dict(child)
            return child
            
        
