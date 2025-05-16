"""
Adapted code from https://github.com/ImperialCollegeLondon/IPWGAN/
"""

import torch
import lightning


class WGANFPN(lightning.LightningModule):
    def __init__(self, netG, netD, wasserstein_criterion, opt, n_levels=4):
        super().__init__()
        self.netG = netG
        self.netD = netD
        self.wasserstein_criterion = wasserstein_criterion
        self.opt = opt
        self.n_levels = n_levels

        # Fixed noise for visualization
        self.gen_iterations = 0
        self.dis_iterations = 0

        self.g_iter_base = 1
        self.dis_iter_interval = 10
        self.gen_iter_interval = 1

        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizerD = torch.optim.RMSprop(self.netD.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        optimizerG = torch.optim.RMSprop(self.netG.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        return [optimizerD, optimizerG]

    def training_step(self, batch, batch_id):
        opt_dis, opt_gen = self.optimizers()
        real_cpu = batch
        batch_size = real_cpu.size(0)

        # Create noise
        noise = torch.FloatTensor(batch_size, self.opt.nz, 1, 1, 1).normal_(0, 1).to(self.device)

        # Only update discriminator every 10 steps as in original code
        if self.dis_iterations % self.dis_iter_interval == 0:
            opt_dis.zero_grad()

            # Train with real
            errD_real = self.netD(real_cpu)

            # Generate fake samples
            fake = self._generate_fake(noise)
            fake = fake.detach()

            errD_fake = self.netD(fake)
            errD = self.wasserstein_criterion(errD_real, errD_fake, real_cpu, fake)
            self.manual_backward(errD)
            opt_dis.step()
            self.log("loss_D", errD, prog_bar=True)

        if self.gen_iterations % self.gen_iter_interval == 0:
            # Generate fake samples
            gen_iter = self.g_iter_base
            while gen_iter != 0:
                opt_gen.zero_grad()
                fake = self._generate_fake(noise)
                pred_fake = self.netD(fake)
                errG = -torch.mean(pred_fake)
                self.manual_backward(errG)
                opt_gen.step()
                gen_iter -= 1
            self.log("loss_G", errG, prog_bar=True)

        self.gen_iterations += 1
        self.dis_iterations += 1

    def _generate_fake(self, noise):
        fake = None
        for level in range(self.n_levels):
            if level == 0:
                fake = self.netG(noise, level)
            else:
                fake = self.netG(noise, level, fake)
        return fake

    def on_epoch_end(self):
        pass
