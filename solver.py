import torch
from tqdm import tqdm
import random
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
scale=0.001
normal = torch.distributions.Normal(loc, scale) # create a normal distribution object
def mutate_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        m.weight.data = m.weight.data + normal.rsample(m.weight.size()).cuda()
      
class Solver:
  def __init__(self, model, optim, loss_fn, val_fn, evo_optim, train, val, test, epochs=100, evo_step=5, child_count=20, best_child_count=3):
    self.model = model
    self.optim = optim
    self.loss_fn = loss_fn
    self.val_fn = val_fn
    self.evo_optim = evo_optim
    self.train = train
    self.val = val
    self.test = test
    self.epochs = epochs
    self.evo_step = evo_step
    self.child_count = child_count
    self.best_child_count = best_child_count

  def start(self):
    print("Start training")
    for epoch in range(self.epochs):
      if (epoch % self.evo_step == 0):
        self.model.eval()
        best_child_score = self.batch("evolve")
        print(f"best child - {best_child_score}%")
      else:
        self.model.train()
        loss, val_score = self.batch("train")
        print('[%d] loss: %.3f validation score: %.2f %%' %
          (epoch + 1, loss, val_score))
    self.model.eval()
    final_score = self.batch("test")
    print('Training is finished\nvalidation score: %.2f %%' %
          (final_score))
    return self.model


  def batch(self, mode):
    if mode == "train":
      loss = 0.0
      for i, data in enumerate(self.train, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        self.optim.zero_grad()

        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        self.optim.step()
      val_score = self.val_fn(self.model, self.val)
      return loss.item(), val_score
    elif mode == "evolve":
      best_kids = MaxSizeList(self.best_child_count)
      best_child = deepcopy(self.model)
      best_child.apply(mutate_weights)
      best_child_score = self.val_fn(best_child, self.val)
      best_kids.push(best_child)
      for _ in range(self.child_count-1):
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
    elif mode == "test":
      return self.val_fn(self.model, self.test)
