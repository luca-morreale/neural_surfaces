
from torch.optim.lr_scheduler import CosineAnnealingLR

class DelayedCosine(CosineAnnealingLR):
    ## start decay after 'start_epoch' has been passed

    def __init__(self, optimizer, T_max, start_epoch=0, eta_min=0, last_epoch=- 1, verbose=False):

        self.start_epoch = start_epoch
        self.past_steps = 0

        super().__init__(optimizer, T_max, eta_min, last_epoch, verbose)


    def step(self):
        self.past_steps += 1

        if self.past_steps >= self.start_epoch and \
                self.past_steps < (self.start_epoch + self.T_max):

            super().step()
