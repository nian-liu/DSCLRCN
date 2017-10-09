% mnist test
clear,clc

addpath('/home/nianliu/Deeplearning_codes/caffe/caffe-master/matlab');
caffe.reset_all();
use_gpu=1;
gpu_id=2;
if use_gpu
  caffe.set_mode_gpu();
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end
model_file='lenet_iter_5000.caffemodel';
model_config='lenet_srnn_global_deploy.prototxt';
net = caffe.Net(model_config, model_file, 'test');

im=imread('mnist_2.png');
im = single(im/255);
net.blobs('data').set_data(im);
net.forward_prefilled();
