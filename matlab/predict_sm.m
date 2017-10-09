%predict salient object saliency maps of a dataset
clear,clc

addpath('/home/nianliu/matlabToolbox');
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

% lamda=0.025;
% model_file='../caffe/models/EF_FCN6_DSLSTM_MIT_finetune_iter_400.caffemodel';
% model_file='../caffe/models/FCN_SLSTM/SALICON/480DSLSTM/EF_FCN6_DSLSTM_iter_5000.caffemodel';
% model_config='../caffe/EF_FCN6_DSLSTM_deploy.prototxt';

scene_context=0;
% allDataset={'SALICON-val'};
allDataset={'MITBenchmark'};
algName='ResNet50_ML_DSCLSTM';

model_file=['../caffe/' algName '/EF_' algName '_best.caffemodel'];
model_config=['../caffe/' algName '/EF_' algName '_deploy.prototxt'];
net = caffe.Net(model_config, model_file, 'test');

mean_pix = [103.939, 116.779, 123.68];
stdsize=[480,640];
%stdsize=[240,320];
saveRootPath='../data/results';
% type={'Gsm','RCL4sm','RCL3sm','RCL2sm','RCL1sm'};

if scene_context
    load('/home/nianliu/Deeplearning_codes/caffe/caffe-master/models/placesCNN/places_mean.mat')
    image_mean=imresize(image_mean,[227,227]);
end

for datasetIdx=1:length(allDataset)
    datasetName=allDataset{datasetIdx};
    disp(datasetName);
    
    datasetInfo=getEFDatasetInfo(datasetName);
    imgPath=datasetInfo.imgPath;
    dmPath=datasetInfo.dmPath;
    imgFiles=datasetInfo.imgFiles;
    %dmFiles=datasetInfo.dmFiles;
    imgNum=datasetInfo.imgNum;
    resultsPath=[saveRootPath '/' datasetName '/'];
    mkdir(resultsPath);
    
    mkdir([resultsPath algName]);
    tic
    for i=1:imgNum
        disp(i)
        %close all
        
        image=imread([imgPath '/' imgFiles(i).name]);
        [imgName,~]=strtok(imgFiles(i).name,'.');
        
        if length(size(image))==2
            image=repmat(image,[1,1,3]);
        end

        im = single(image);
        
%         siz=[size(image,1),size(image,2)];
%         [minSiz,idx]=min(siz);
%         if idx==1
% 		    sizTmp=[480, round(siz(2)*480/minSiz)];
%             im = imresize(im, sizTmp);
%         else
% 		    sizTmp=[round(siz(1)*480/minSiz),480];
%             im = imresize(im, sizTmp);
% 		end
        
        im = imresize(im, stdsize);
        im = im(:, :, [3 2 1]);
        im = permute(im, [2 1 3]);
        for c = 1:3
            im(:, :, c) = im(:, :, c) - mean_pix(c);
        end
        
        if scene_context
            im1 = single(image);
            im1 = imresize(im1, [227,227]);
            im1 = im1(:, :, [3 2 1]);
            im1 = permute(im1, [2 1 3]);
            im1=im1-image_mean;
            net.blobs('227img').set_data(im1);
        end
        
        net.blobs('img').set_data(im);
        net.forward_prefilled();
		output=net.blobs('sm').get_data();
		
% 		if idx==1
% 		    output=reshape(output,[],60);
%         else
% 		    output=reshape(output,60,[]);
%         end
        
        
% 		output=reshape(output,stdsize(end:-1:1)/8);
        %results{1}=normalize(imresize(net.blobs('upsampledSm').get_data()',[size(image,1),size(image,2)]));
		sm=imresize(output',[size(image,1),size(image,2)]);
        
%         sigma=minSiz*lamda; 
%         g = fspecial('gaussian', double(uint8(round([sigma sigma]*4))/2*2+1), sigma);
%         output=imfilter(output,g,'replicate');%sm gaussian smoothed
        results=normalize(sm);
        
        eval(['imwrite( results, [resultsPath  ''' algName ''' ''/'' ''' imgName ''' ''.jpg'']);']);
    end
    t=toc;
    avgT=t/imgNum
end
caffe.reset_all();
% matlabpool close