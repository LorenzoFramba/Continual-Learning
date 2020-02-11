import torch
import torch.nn as nn

'''
Append padding to keep concat size & input-output size
'''

class DownBlock(nn.Module):     #downBlock
    def __init__(self, in_dim, out_dim):
        super(DownBlock, self).__init__()
        self.block = nn.Sequential(   #i moduli verranno aggiunti in modo sequenziale al modello
            nn.MaxPool2d(kernel_size=2, stride=2),      #MaxPool diminuiamo la dimensione, prendendo il valore max in una matrice 2x2. quindi la dimensione sara' di un quarto dell'originale ( penso)
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1), #in_dim e' il numero di canali RGB, cioe' 3 . 
            nn.ReLU(),  #elimininiamo i risultati negativi
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1), #convoluzione da fine a fine
            nn.ReLU())  #toglie i negativi

    def forward(self, x):
        out = self.block(x)  
        return out

class UpBlock(nn.Module):  #upBlock
    def __init__(self, in_dim, mid_dim, out_dim):
        super(UpBlock, self).__init__()  #inizializza il modulo interno
        self.block = nn.Sequential(   #i moduli verranno aggiunti in modo sequenziale al modello
            nn.Conv2d(in_dim, mid_dim, kernel_size=3, stride=1, padding=1),  #convolution da inizio a meta
            nn.ReLU(),          #toglie i degativi
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1), #convoluzione da meta a meta 
            nn.ReLU(),                  #toglie i negativi
            nn.ConvTranspose2d(mid_dim, out_dim, kernel_size=2, stride=2)) #convoluzione finale da meta a fine

    def forward(self, x):
        out = self.block(x)
        return out

class UNet(nn.Module):
    def __init__(self, num_classes, in_dim=3, conv_dim=64):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.conv_dim = conv_dim
        self.build_unet()

    def build_unet(self):
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.conv_dim, kernel_size=3, stride=1, padding=1),   # prima convoluzione
            nn.ReLU(),                                  #toglie i negativi
            nn.Conv2d(self.conv_dim, self.conv_dim, kernel_size=3, stride=1, padding=1), #seconda convoluzione
            nn.ReLU(),                                  #toglie i negativi
        self.enc2 = DownBlock(self.conv_dim, self.conv_dim*2)  #va in giu, aumenta convoluzione a 128. input da 64, output a 128 
        self.enc3 = DownBlock(self.conv_dim*2, self.conv_dim*4) #va in giu, aumenta convoluzione a 256. input da 128, output a 256
        self.enc4 = DownBlock(self.conv_dim*4, self.conv_dim*8) #va in giu, aumenta convoluzione a 512. input da 256, output a 512

        self.dec1 = UpBlock(self.conv_dim*8, self.conv_dim*16, self.conv_dim*8)    #torna in su da 512 a 1024 a 512
        self.dec2 = UpBlock(self.conv_dim*16, self.conv_dim*8, self.conv_dim*4) #torna in su da da 1024 512 a 256
        self.dec3 = UpBlock(self.conv_dim*8, self.conv_dim*4, self.conv_dim*2)  #torna in su da 512 a  256 a 128
        self.dec4 = UpBlock(self.conv_dim*4, self.conv_dim*2, self.conv_dim)    #torna in su da 256 a 128 a 64

        self.last = nn.Sequential(                          #i moduli verranno aggiunti in modo sequenziale al modello
            nn.Conv2d(self.conv_dim*2, self.conv_dim, kernel_size=3, stride=1, padding=1),      #conv da 128 a 64
            nn.ReLU(),              #solo positivi
            nn.Conv2d(self.conv_dim, self.conv_dim, kernel_size=3, stride=1, padding=1), #conv da 64 a 64
            nn.ReLU(),                  #solo positivi
            nn.Conv2d(self.conv_dim, self.num_classes, kernel_size=1, stride=1)) #conv da 64 a 64

    def forward(self, x):
        enc1 = self.enc1(x) # 16
        enc2 = self.enc2(enc1) # 8
        enc3 = self.enc3(enc2) # 4
        enc4 = self.enc4(enc3) # 2

        center = nn.MaxPool2d(kernel_size=2, stride=2)(enc4)

        dec1 = self.dec1(center) # 4
        dec2 = self.dec2(torch.cat([enc4, dec1], dim=1))
        dec3 = self.dec3(torch.cat([enc3, dec2], dim=1))
        dec4 = self.dec4(torch.cat([enc2, dec3], dim=1))

        last = self.last(torch.cat([enc1, dec4], dim=1))
        assert x.size(-1) == last.size(-1), 'input size(W)-{} mismatches with output size(W)-{}' \
                                            .format(x.size(-1), output.size(-1))
        assert x.size(-2) == last.size(-2), 'input size(H)-{} mismatches with output size(H)-{}' \
                                            .format(x.size(-1), output.size(-1))
        return last

if __name__ == '__main__':
    sample = torch.randn((2, 3, 32, 32))
    model = UNet(num_classes=2)
    print(model(sample).size())
