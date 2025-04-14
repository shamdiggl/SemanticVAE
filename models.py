import torch
import torch.nn as nn
import torch.nn.functional as F
class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # First conv and norm
        self.norm1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        
        # Second conv and norm 
        self.norm2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        
        
        self.relu = nn.ReLU()
        
        # `in_channels` to `out_channels` mapping layer for residual connection
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        else:
            self.projection = nn.Identity()
            
            
    def forward(self, x: torch.Tensor):
        h = x

        # First normalization and convolution layer
        h = self.norm1(h)
        h = self.relu(h)
        h = self.conv1(h)


        # Second normalization and convolution layer
        h = self.norm2(h)
        h = self.relu(h)
        h = self.conv2(h)
      
        # Map and add residual
        return self.projection(x) + h
        
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1), # Output: [batch_size, 16, 14, 14]
            ResnetBlock(16,16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1), # Output: [batch_size, 16, 7, 7]
            ResnetBlock(16,32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1), # Output: [batch_size, 32, 4, 4]
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*4*4, 64), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 10),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class ResNet_AutoEncoder(nn.Module):
    def __init__(self, channels: list, num_of_blocks: int, down_sample: list, latent_dim: int):
        super(ResNet_AutoEncoder, self).__init__()
        self.channels = channels
        self.encoder = nn.Sequential()
                
        self.encoder.append(nn.Conv2d(in_channels=1, out_channels=channels[0], 
                            kernel_size=3, stride=1, padding=1))
        
        for i in range(len(channels)-1): #ResNet layers
            in_channel = channels[i]
            out_channel = channels[i+1]        
            for j in range(num_of_blocks):
                self.encoder.append(ResnetBlock(in_channel,out_channel))
                in_channel = out_channel
            if down_sample[i] == True:
                self.encoder.append(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, 
                      kernel_size=3, stride=2, padding=1))
        
        
        self.mu = nn.Linear(channels[-1]*4, latent_dim)
        self.logvar = nn.Linear(channels[-1]*4, latent_dim)
        
       
        self.decoder_in = nn.Linear(latent_dim, channels[-1]*4)
        self.decoder = nn.Sequential()           
        
        for i in range(1,len(channels)): #ResNet layers
            in_channel = channels[-i]
            out_channel = channels[-i-1] 
            if down_sample[-i] == True:
                self.decoder.append(nn.ConvTranspose2d(in_channels=in_channel, out_channels=in_channel, 
                                    kernel_size=3, stride=2, padding=1, output_padding=1))
            for j in range(num_of_blocks):
                self.decoder.append(ResnetBlock(in_channel,out_channel))
                in_channel = out_channel
                
        self.decoder.append(nn.Conv2d(in_channels=channels[0], out_channels=1, kernel_size=3, stride=1, padding=1))            
        self.decoder.append(nn.Tanh())
    
    def reparameterize(self, mu, logvar):
        # Reparameterization trick to sample from N(mu, var)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    
    def encode(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0),-1)
        return encoded
    
    def decode(self, z):
        z = self.decoder_in(z)
        z = z.view(z.size(0),self.channels[-1],2,2)
        decoded = self.decoder(z)[:,:,:28,:28]
        return decoded
    
    def forward(self, x):
        encoded = self.encode(x)
        
        mu = self.mu(encoded)        
        logvar = self.logvar(encoded)        
        z = self.reparameterize(mu,logvar)
        
        decoded = self.decode(z)
        return decoded, z, mu, logvar