"""
Training code for Adversarial patch training


"""

import PIL
import load_data
from tqdm import tqdm

from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess

from torchvision.datasets import CocoDetection

import patch_config
import sys
import time

class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()

        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
        self.patch_applier = PatchApplier().cuda()
        # self.patch_transformer = PatchTransformer().cuda()
        self.patch_transformer = PatchTransformerKeypointsTensors().cuda()
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        self.prob_extractor_class = MaxProbExtractor(15, 80, self.config).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

        self.writer = self.init_tensorboard(mode)

    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        img_size = self.darknet_model.height
        batch_size = self.config.batch_size
        n_epochs = 100
        max_lab = 14

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate starting point
        adv_patch_cpu = self.generate_patch("gray")
        #adv_patch_cpu = self.read_image("@PJ_PATCHES/patch_scale1_5.jpg")

        adv_patch_cpu.requires_grad_(True)

        # train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
        #                                                         shuffle=True),
        #                                            batch_size=batch_size,
        #                                            shuffle=True,
        #                                            num_workers=10)

        train_loader = torch.utils.data.DataLoader(CocoKeypointDataset(self.config.img_dir, self.config.lab_dir, img_size, max_lab),
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=10)
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        aant_show = 10

        et0 = time.time()
        for epoch in range(n_epochs):
            tellerke = 0
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_class_loss = 0
            ep_loss = 0
            bt0 = time.time()
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                with autograd.detect_anomaly():
                    # print(lab_batch[1, :])
                    # img = img_batch[0, :, :,]
                    # im = transforms.ToPILImage('RGB')(img)
                    # plt.figure(300)
                    # plt.imshow(im)
                    # exit()

                    img_batch = img_batch.cuda()
                    lab_batch = torch.FloatTensor(lab_batch).cuda()

                    #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()
                    # adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))

                    # print(lab_batch[1, :])
                    # img = p_img_batch[0, :, :,]
                    # im = transforms.ToPILImage('RGB')(img.detach().cpu())
                    # plt.figure(400)
                    # plt.imshow(im)
                    # plt.show()
                    # exit()

                    img = p_img_batch[0, :, :,]
                    img = transforms.ToPILImage()(img.detach().cpu())

                    if(tellerke < aant_show):
                        img.save(f"@PJ_PATCHES/test_tensor{tellerke}.png")
                        tellerke += 1

                    output = self.darknet_model(p_img_batch)
                    max_prob = self.prob_extractor(output)
                    class_prob = self.prob_extractor_class(output)
                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)

                    nps_loss = nps*0.005
                    tv_loss = tv*0.75
                    det_loss = torch.mean(max_prob)
                    class_loss = (1-torch.mean(class_prob))*0.5
                    loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda()) + class_loss

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_class_loss += class_loss.detach().cpu().numpy()
                    ep_loss += loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1)       #keep patch in image range

                    bt1 = time.time()
                    if i_batch%5 == 0:
                        iteration = self.epoch_length * epoch + i_batch

                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/class_loss', class_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                        self.writer.add_image('patch', adv_patch_cpu, iteration)
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss, class_loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()

            # if epoch%5 == 0:
            #     #img.show()
            #     img.save("@PJ_PATCHES/patch_cat2_scale1_5_persons.png")

            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_class_loss = ep_class_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)

            #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            #plt.imshow(im)
            #plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('CLASS LOSS: ', ep_class_loss)
                print('EPOCH TIME: ', et1-et0)
                im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                #plt.imshow(im)
                #plt.show()
                im.save("@PJ_PATCHES/final_cat_1s.jpg")
                del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss, class_loss
                torch.cuda.empty_cache()
            et0 = time.time()



    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)

    trainer = PatchTrainer(sys.argv[1])
    trainer.train()

if __name__ == '__main__':
    main()
