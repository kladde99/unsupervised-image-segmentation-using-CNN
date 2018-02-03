close all;
clear all;
clc;
% addpath(genpath('/net/linse8/no_backup_01/s1184'));

cd /home/s1184/matlab/edition_test/test2segpcamerge/;
run('/home/s1184/matlab/matconvnet-new-solvers/matlab/vl_setupnn');
load('/home/s1184/matlab/bsds500works/test2/net-epoch-900.mat');
load('/home/s1184/matlab/edition_test/imdb_test.mat');
load('/home/s1184/matlab/edition_test/TestImageIdx.mat');

net.layers(end) = [];
segs = cell(0);

for id= 1:10

Imname =[files(id).name(1:end-4) '.mat'];
TestIm = im2single(imdb3.images{id});
[Imrow,Imcol,Imdepth] = size(TestIm);


% SmoothIm = uint8(vl_imsmooth(TestIm,2));
sign = 0;
if Imrow > Imcol
TestIm = permute(TestIm,[2 1 3]);
sign = 1;
end
%    TestIm = imresize(TestIm,[128 192]);
res = vl_simplenn(net, TestIm) ;

%% RGB+features from cae trained layer
% remove the zero matrix
idx =[];
for i = 1: size(res(9).x,3)
imdetect = res(9).x(:,:,i);
if ~any(imdetect(:))
   idx =[idx,i];
end
end

res(9).x(:,:,idx)=[];

code_layer = res(9).x;

if sign ==1
 code_layer = permute(code_layer,[2 1 3]);
 TestIm = permute(TestIm,[2 1 3]);
end


code_layer_transfer = imresize(code_layer,[Imrow Imcol]);
% code_layer = code_layer(:);
% code_layer = (code_layer-min(code_layer))/(max(code_layer)-min(code_layer))*255;
% cae_features_rgb2 = uint8(reshape(code_layer,Imrow,Imcol,[]));

% dimension reduction

% NewChannels = cat(3,TestIm, code_layer_transfer);
NewChannels = code_layer_transfer;

%  normalization
NewChannels_vector = NewChannels(:);
NewChannels_norm = (NewChannels_vector-min(NewChannels_vector))/...
             (max(NewChannels_vector)-min(NewChannels_vector));

FeatureVector = reshape(NewChannels_norm,size(NewChannels,1)*size(NewChannels,2),...
            size(NewChannels,3));
[coeff,score,latent] = pca(FeatureVector);
TransferIm = FeatureVector*coeff;

% norm
% TransferIm_vector = TransferIm(:);
% TransferIm_norm = (TransferIm_vector-min(TransferIm_vector))/...
%                  (max(TransferIm_vector)-min(TransferIm_vector));

TransferIm_new = im2uint8(reshape(TransferIm,size(NewChannels,1),size(NewChannels,2),...
         size(NewChannels,3)));

DimReductionIm  = TransferIm_new (:,:,1:3);
% DimReductionIm  = TransferIm_new (:,:,1);

figure(id);clf;
subplot(1,2,1); imagesc(TestIm);
colormap gray; axis image off;
title('original image');
subplot(1,2,2); imagesc(DimReductionIm);
colormap gray; axis image off;
title('pca merge 3 channels');
myfilename=['id_',num2str(id),'_pcaMerge3channels','.jpg'];
saveas(gcf,myfilename);

%% parameter evaluation
%  nParameter = 11;
%  KernelSize = 2:4:42;
%  parameter1 = repmat(KernelSize,1,nParameter);
%  parameter2 = repmat(KernelSize',1,nParameter)';
%  parameter2 = parameter2(:)';
%  KernelParamater = [parameter1;parameter2];

%  parfor k= 1:nParameter^2

SpatialBW  = 20; RangeBW  = 16;

[labels,modes,regSize] = edison_wrapper(DimReductionIm , @RGB2Luv,...
'SpatialBandWidth',SpatialBW,...
'RangeBandWidth',RangeBW);





%%imresize
SegIm =rgb2gray (uint16(labels));
[Segrow Segcol]= size(SegIm);

if Segrow > Segcol
SegIm_new = imresize(SegIm,[481 321]);
else
SegIm_new = imresize(SegIm,[321 481]);
end
[SegrowNew SegcolNew]= size(SegIm_new);

OldPixelArray = SegIm_new(:);
NewPixelArray = OldPixelArray;
SortOldPixelArray=sort(unique(OldPixelArray));
ncluster=length(SortOldPixelArray);

for i=1:ncluster
idx=find(OldPixelArray==SortOldPixelArray(i));
NewPixelArray(idx)=i;
end
segs = reshape(NewPixelArray,SegrowNew,SegcolNew);

%% plot
figure(10*id);
subplot(1,2,1);
imagesc(imdb3.images{id});axis image off;
title('original image');
subplot(1,2,2);
imagesc(label2rgb(segs));axis image off;colormap gray;
title('seg using pca features');
myfilename=['id_',num2str(id),'_segUsingPCAfeatures','.jpg'];
saveas(gcf,myfilename);

end;