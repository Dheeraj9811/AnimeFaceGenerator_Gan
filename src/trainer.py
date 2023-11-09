from IPython.display import clear_output
from utils import register, images
from tqdm import tqdm
import numpy as np
import torch
import warnings
import wandb
import time
import subprocess
import psutil


def is_gpu_available():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv,nounits,noheader'])
        return True
    except FileNotFoundError:
        return False
# Function to get GPU memory usage using nvidia-smi
def get_gpu_memory():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
        memory_used = [int(x) // 1048576 for x in result.decode('utf-8').strip().split('\n')]  # Convert bytes to MB
        print(f"GPU Memory Used: {memory_used}")
        return memory_used
    except FileNotFoundError:
        return []

trainers = register.ClassRegistry()


@trainers.add_to_registry(name="base")
class BaseGANTrainer:
    def __init__(self, conf, **kwargs):
        '''
        Input:
            conf: dictionary of training settings
            **kwargs: the dictionary must contain:
                "G": initialized generator
                "D": initialized discriminator
                "start_epoch": continuing training
                "dataloader": dataset loader
                "optim_G": generator optimizer
                "optim_D": discriminator optimizer
                "gen_loss": generator loss function
                "disc_loss":  discriminator loss function
                "z_dim": the dimension of the generator vector
                "device": contains a device type
        '''
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.conf = conf
        if self.conf['wandb_set']:
            wandb.login()
            wandb.init(
                project=self.conf['Generator'],
                config=self.conf
            )
            wandb.watch(self.G)
            wandb.watch(self.D)
          

    def logger(self, data):
        
        if self.conf['wandb_set']:
            wandb.log(data)
        
  
    def save_model(self, epoch):
        state = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'start_epoch': epoch + 1,
        }
        torch.save(state, f'{self.conf["Weight_dir"]}/weight {epoch + 1}.pth')


    def generate_images(self, cnt=1):
        # Sample noise as generator input
        z = torch.randn((cnt, self.z_dim), device=self.device)
        return self.G(z)
        
    
    def train_disc(self, real_imgs, fake_imgs):
        real_logits = self.D(real_imgs)
        fake_logits = self.D(fake_imgs)
        return self.disc_loss(real_logits, fake_logits)
        
        
    def train_gen(self, fake_imgs):
        logits_fake = self.D(fake_imgs)
        return self.gen_loss(logits_fake)
    
    
    def train_loop(self):
        start_time = []
        ram_memory_before = psutil.virtual_memory().used // 1048576  # Convert bytes to MB
        gpu_memory_before = get_gpu_memory() if is_gpu_available() else []
        
        warnings.filterwarnings("ignore")
        for epoch in range(self.start_epoch, self.conf['epochs']):
            start_time.append(time.time())
            bar = tqdm(self.dataloader)
            loss_G, loss_D = [], []
            for i, real_img in enumerate(bar):
                self.G.zero_grad()
                real_imgs = real_img.to(self.device)
                
                # Generate a batch of images
                fake_imgs = self.generate_images(real_imgs.size(0)).detach()

                # Update D network
                d_loss = self.train_disc(real_imgs, fake_imgs)
                
                self.D.zero_grad()
                d_loss.backward()
                self.optim_D.step()

                loss_D.append(d_loss.item())
                self.logger({"loss_D":d_loss.item()})

                if i % self.conf['UPD_FOR_GEN'] == 0:
                    # Generate a batch of images
                    fake_imgs = self.generate_images(real_imgs.size(0))

                    # Update G network
                    g_loss = self.train_gen(fake_imgs)
                    
                    self.G.zero_grad()
                    g_loss.backward()
                    self.optim_G.step()

                    loss_G.append(g_loss.item())
                    self.logger({"loss_G":g_loss.item()})

                # Output training stats
                if i % 5 == 0:
                    clear_output(wait=True)
                    with torch.no_grad():
                        self.G.eval()
                        Image = images.TensorToImage(self.generate_images().detach().cpu()[0], 0.5, 0.225)
                        # self.logger({"Random generated face": wandb.Image(Image)})
                        self.G.train()

                clear_output(wait=True)        
                bar.set_description(f"Epoch {epoch + 1}/{self.conf['epochs']} D_loss: {round(loss_D[-1], 2)} G_loss: {round(loss_G[-1], 2)}")


            self.logger({"mean_loss_G":np.mean(loss_G), "mean_loss_D":np.mean(loss_D), "time":{time.time() - start_time[-1]}})
            print(f"Epoch {epoch + 1}/{self.conf['epochs']} D_loss: {round(np.mean(loss_D), 2)} G_loss: {round(np.mean(loss_G), 2)}")
        
        # Save model
        self.save_model(self.conf['epochs'])

        ram_memory_after = psutil.virtual_memory().used // 1048576  # Convert bytes to MB
        gpu_memory_after = get_gpu_memory() if is_gpu_available() else []

        # Calculate the memory used by your function
        ram_memory_used = ram_memory_after - ram_memory_before
        gpu_memory_used = [gpu_after - gpu_before for gpu_before, gpu_after in zip(gpu_memory_before, gpu_memory_after)]
        
        with open('logs.txt', 'a') as file:
            file.write(f"RAM Memory Used: {ram_memory_used} MB\n")
            file.write(f"Total Time taken: {sum(start_time)}\n")
        
        if gpu_memory_used:
            with open('logs.txt', 'a') as file:
                file.write(f"GPU Memory Used: {gpu_memory_used} MB\n")


@trainers.add_to_registry(name="gp")            
class GpGANTrainer(BaseGANTrainer):
    def __init__(self, conf, **kwargs):
        super().__init__(conf, **kwargs) 
        
        
    def train_disc(self, real_imgs, fake_imgs):
        lambda_gp = self.conf["Loss_config"]["lambda_gp"]
        return self.disc_loss(self.D, real_imgs, fake_imgs, lambda_gp)
