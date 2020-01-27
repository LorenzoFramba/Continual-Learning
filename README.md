## Continual Learning without Forgetting - Semantic Segmentation

pytorch implementation of Semantic segmentation using Unet as the network. 

### Some results 
***Pascal VOC 2012 / U-Net***
- Input -> Ground Truth -> Generated
<p align='center'>  
  <img src='preview.gif' />
</p>

### Currnet
- U-Net

### Some details

* I applied different metric algorithms .
* I used augmented dataset. (Currently, I only applied flip operation for augmentation.)


## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install torch torchvision and visdom


```bash
pip3 install torch torchvision
pip3 install visdom
```

- Clone this repo:
```bash
git clone https://github.com/LorenzoFramba/Continual-Learning.git
cd Continual-Learning
```

- Import torch and install cuda
```bash
import torch 
torch.cuda.is_available()
```
**To train models**

```bash
python main.py --mode train --model unet --dataset voc \
-- n_iters 10000 --train_batch_size 16 val_batch_size 16 \
--h_image_size 256 --w_image_size 256 \
--model_save_path './models' --sample_save_path './samples'
```

**To load the model**
```bash
--path ../path
--which_epoch 'latest' \
--continue_train \
```

### Dependencies
* [python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.1](http://pytorch.org/)




* **Fully Convolutional Networks for Semantic Segmentation**[\[paper\]](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) <br/>
  *Jonathan Long, Evan Shelhamer, Trevor Darrell*
* **U-Net: Convolutional Networks for Biomedical Image Segmentation** [\[paper\]](https://arxiv.org/abs/1505.04597) <br/>
  *Olaf Ronneberger, Philipp Fischer, Thomas Brox*
* **Continual Learning for Dense Labeling of Satellite Images** [\[paper\]](https://hal.inria.fr/hal-02276543/document) <br/>
  *Onur Tasar, Yuliya Tarabalka, Pierre Alliez*
* **Learning a Discriminative Feature Network for Semantic Segmentation** [\[paper\]](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0632.pdf) <br/>
  *Changqian Yu, Jingbo Wang, Chao Peng, Changxin Gao, Gang Yu, Nong Sang*
  
  
  
### Contact
*If you have any questions about codes, let me know! .*
