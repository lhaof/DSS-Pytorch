# DSS-Pytorch

This code is a pytorch implementation of 'Deeply Supervised Salient Object Detection with Short Connections' (DSS, CVPR2017, PAMI2018). If you use this code, you may consider citing the following paper. By the way, the code is implemented by lhaof. You may also cite the github link of this repository.
```
@article{hou2017deeply,
  title={Deeply Supervised Salient Object Detection with Short Connections},
  author={Hou, Qibin and Cheng, Mingming and Hu, Xiaowei and Borji, Ali and Tu, Zhuowen and Torr, Philip H S},
  journal={computer vision and pattern recognition},
  volume={41},
  number={4},
  pages={5300--5309},
  year={2017}
}
```
Step 1:
```
Download VGG_ILSVRC_16_layers.caffemodel & VGG_ILSVRC_16_layers_deploy.prototxt
(I believe you can find them easily.)
```
Step 2:
```
Download the source code of Caffe and install it. To find a Caffe code, you may 
search with keywords 'caffe-dss github'.
```
Step 3:
```
Run load_vgg_caffe_to_pytorch.py and convert VGG_*.caffemodel into vgg16.pth.
```
Step 4:
```
Have a look at dataloaders. Prepare the datasets at the corresponding directory.
```
Step 5:
```
python train.py
```
