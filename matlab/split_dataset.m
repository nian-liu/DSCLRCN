%split MIT dataset into train set and test set
function split_dataset

clear,clc
addpath('/home/nianliu/matlabToolbox');

imgSiz=[480,640];
fixSiz=[480,640];
Datasets={'MIT'};
SplitNums=[903,100];
set={'train','val'};
      
splitDir='./';

trainImgs=cell(1806,1);
trainFms=cell(1806,1);
valImgs={};
valFms={};
cellIdx=0;
for DatasetIdx=1:length(Datasets)
    DatasetName=Datasets{DatasetIdx};
    disp(DatasetName);
    newPath=[splitDir DatasetName '/'];
    datasetInfo=getEFDatasetInfo(DatasetName);
    imgPath=datasetInfo.imgPath;
    fmPath=datasetInfo.fmPath;
	dmPath=datasetInfo.dmPath;
    imgFiles=datasetInfo.imgFiles;
    imgNum=datasetInfo.imgNum;
    SplitNum=SplitNums(DatasetIdx,:);
    SplitNum=[0,SplitNum];
	if exist([newPath 'idx-' DatasetName '.mat'], 'file')
        load([newPath 'idx-' DatasetName '.mat']);
	else
	    idx=randperm(imgNum);
		save([newPath 'idx-' DatasetName '.mat'],'idx');
	end
    for i=1:imgNum
        disp(i);
        [name,~]=strtok(imgFiles(idx(i)).name,'.');
        for j=1:numel(SplitNum)-1
            if i>sum(SplitNum(1:j)) && i<=sum(SplitNum(1:j+1))
               break
            end
        end
        thisSet=set{j};
%         copyfile([imgPath '/' imgFiles(idx(i)).name],[newPath thisSet '/images']);
%         copyfile([fmPath '/' name '.mat'],[newPath thisSet '/fixation']);
% 		copyfile([dmPath '/' name '.jpg'],[newPath thisSet '/density_maps']);
% 		im=imread([imgPath '/' imgFiles(idx(i)).name]);
% 		im=imresize(im,imgSiz);
		fixData=load([fmPath '/' name '.mat']);
        fixVarName = fieldnames(fixData);
        Fix = getfield(fixData,char(fixVarName));
        Fix = im2double(Fix);
        Fix = resizeFix(Fix,fixSiz);
%         copyfile([datasetSplitDir 'img/' DatasetName '_' name '*.tif'],['../img_fm/' thisSet '/img']);
%         for j=1:length(fmDir)
%             copyfile([datasetSplitDir fmDir{j} '/' DatasetName '_' name '*.png'],['../img_fm/' thisSet '/' fmDir{j}]);
%         end
    switch thisSet
        case 'train'
% 		    imwrite(im,[newPath thisSet '/CNN_images/' name '-1.jpg']);
% 			imwrite(im(:,end:-1:1,:),[newPath thisSet '/CNN_images/' name '-2.jpg']);
			imwrite(Fix,[newPath thisSet '/CNN_fixation/' name '-1.png']);
			imwrite(Fix(:,end:-1:1,:),[newPath thisSet '/CNN_fixation/' name '-2.png']);
%             for j=1:2
%                 cellIdx=cellIdx+1;
% 				nameTmp=[name '-' num2str(j)];
%                 fulName=[thisSet '/CNN_images/' nameTmp '.jpg'];
%                 trainImgs{cellIdx}=fulName;
%                 fmName=[thisSet '/CNN_fixation/' nameTmp '.png'];
%                 trainFms{cellIdx}=fmName;
%             end
        case 'val'
% 		    imwrite(im,[newPath thisSet '/CNN_images/' name '.jpg']);
			imwrite(Fix,[newPath thisSet '/CNN_fixation/' name '.png']);
%             fulName=[thisSet '/CNN_images/' name '.jpg'];
%             valImgs=[valImgs;fulName];
%             fmName=[thisSet '/CNN_fixation/' name '.png'];
%             valFms=[valFms;fmName];
    end
    end
end

%{
trainNum=length(trainImgs);
idx=randperm(trainNum);
fp1=fopen(['train_img_list.txt'],'w');
fp2=fopen(['train_fix_list.txt'],'w');
for i=1:trainNum
    fprintf(fp1,'%s %d\n',trainImgs{idx(i)}, 1);
	fprintf(fp2,'%s %d\n',trainFms{idx(i)}, 1);
end
fclose(fp1);
fclose(fp2);

valNum=length(valImgs);
fp1=fopen(['val_img_list.txt'],'w');
fp2=fopen(['val_fix_list.txt'],'w');
for i=1:valNum
    fprintf(fp1,'%s %d\n',valImgs{i}, 1);
    fprintf(fp2,'%s %d\n',valFms{i}, 1);
end
fclose(fp1);
fclose(fp2);
%}

function resizedFix=resizeFix(Fix,fixSiz)
orgSiz=[size(Fix,1),size(Fix,2)];
if fixSiz(1)~=orgSiz(1) || fixSiz(2)~=orgSiz(2)
    resizedFix=zeros(fixSiz);
    [x,y]=find(Fix==1);
    for i=1:numel(x)
        x1=max(1,min(round(x(i)*fixSiz(1)/orgSiz(1)),fixSiz(1)));
        y1=max(1,min(round(y(i)*fixSiz(2)/orgSiz(2)),fixSiz(2)));
        resizedFix(x1,y1)=1;
    end
else
    resizedFix=Fix;
end