import sys
caffe_root = './caffe_dss/'
sys.path.insert(0, caffe_root+'python')
import caffe 
import numpy as np
from vggnet import vgg16net
import torch

# np.set_printoptions(threshold='nan')
model_file = '../pre_train/VGG_ILSVRC_16_layers_deploy.prototxt'
pretrain_file = '../pre_train/VGG_ILSVRC_16_layers.caffemodel'

# params_txt = './pretrained_caffemodels/params.txt'  
# pf = open(params_txt, 'w')

caffe_net = caffe.Net(model_file, pretrain_file, caffe.TEST) 

pytorch_net = vgg16net()

param_list = list()
param_name_list = list()

for param_name in caffe_net.params.keys():
    weight = caffe_net.params[param_name][0].data
    bias = caffe_net.params[param_name][1].data

    param_list.append(torch.from_numpy(weight))
    param_list.append(torch.from_numpy(bias))

    # print(param_name)
    param_name_list.append(param_name + '.weight')
    param_name_list.append(param_name + '.bias')

net_kvpair = pytorch_net.state_dict()
count = 0
for key, value in net_kvpair.items():
    print('Pytorch key: %s Caffe key: %s'%(key, param_name_list[count]))
    print('Pytorch key.size: ',net_kvpair[key].size(),' Caffe key.size: ',param_list[count].size())
    assert(net_kvpair[key].size() == param_list[count].size())
    net_kvpair[key] = param_list[count]
    count += 1

torch.save(net_kvpair, 'vgg16.pth')

# print(net_kvpair['conv_fuse.weight'])

# torch.save(net_kvpair, './pretrained_caffemodels/dss.pth')


#     print(param_name+'.weight: ' + str(weight.shape) + str(weight.dtype))
#     print(param_name+'.bias: ' + str(bias.shape) + str(bias.dtype))

#     pf.write(param_name)
#     pf.write('\n')

#     pf.write('\n' + param_name + '_weight:\n\n')
#     weight.shape = (-1, 1)

#     for w in weight:
#         pf.write('%ff, ' % w)

#     pf.write('\n\n' + param_name + '_bias:\n\n')

#     bias.shape = (-1, 1)

#     for b in bias:
#         pf.write('%ff, ' % b)

#     pf.write('\n\n') 
# pf.close()

