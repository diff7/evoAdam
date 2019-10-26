import itertools 


class CrossN:
    def __init__(self, params, n_combinations = 2):
        self.params = params
        self.n_combinations = n_combinations
        
    def make_ancestry(self, nets):
        n = []
        for counter, net in enumerate(nets):
            if not 'ancestry' in net.__dir__():
                net.ancestry = [str(counter)]
            n.append(net)
        return n 
 
    def make_combinations(self, nets):
        nets = self.make_ancestry(nets)
        combos = dict()
        for net in nets:
            combos[net.ancestry[-1]] = net
        new_pairs = itertools.combinations(combos.keys(),
                                           min(len(combos.keys()), 
                                               self.n_combinations))
        return new_pairs
