
from tqdm import trange

from .training_loop_gradient import GradientTrainingLoop


class ModulatingTrainingLoop(GradientTrainingLoop):
    ### used in Neural Convolutional Surfaces

    def modulate(self):
        ## change learning rate of optimizers
        ## needed to optimize properly coarse and fine networks
        next_lr = self.target_lr - self.optimizers[0].param_groups[0]['lr']
        self.optimizers[1].param_groups[0]['lr'] = next_lr

        _lambda = next_lr / self.target_lr
        return _lambda


    def loop(self):

        _lambda = 0.0
        it = 0

        self.optimizers[1].param_groups[0]['lr'] = 1.0e-4

        for epoch in trange(self.num_epochs):

            for batch in self.train_loader:

                self.zero_grads()

                batch = self.runner.move_to_device(batch)
                batch['lambda'] = _lambda
                loss, logs = self.runner.train_step(batch)

                loss.backward()

                if it < self.it_start_transition:
                    self.optimizers[0].step()

                elif it >= self.it_end_transition:
                    self.optimizers[1].param_groups[0]['lr'] = self.target_lr
                    self.optimize()
                    _lambda = 1.0
                    self.scheduling()

                else:
                    self.optimize()
                    self.scheduling()
                    _lambda = self.modulate()


                logs['lambda'] = _lambda
                self.log_train(logs)


                it += 1


            self.checkpointing(epoch)

    def optimize(self):
        for opt in self.optimizers:
            opt.step()
