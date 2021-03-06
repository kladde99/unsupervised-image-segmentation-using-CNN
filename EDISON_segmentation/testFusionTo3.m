%% PCA of features to three dimensions
close all;
clear all;
clc;
%% load data,toolbox and the network we have trained
run('../quellcode/matconvnet-new-solvers/matlab/vl_setupnn');
load('../quellcode/Training_BSDS500/data/test/net-cae1.mat');
load('../quellcode/Training_BSDS500/imdb_test.mat');
load('../quellcode/Training_BSDS500/TestImageIdx.mat');
m=9; % index of the code layer 

net.layers(end) = [];
segs = cell(0);
tic;
for id=1:200

Imname =[files(id).name(1:end-4) '.mat'];
TestIm = im2single(imdb3.images{id});
[Imrow,Imcol,Imdepth] = size(TestIm);

sign = 0;
if Imrow > Imcol
TestIm = permute(TestIm,[2 1 3]);
sign = 1;
end

res = vl_simplenn(net, TestIm) ;

code_layer = res(m).x;

if sign ==1
code_layer = permute(code_layer,[2 1 3]);
TestIm = permute(TestIm,[2 1 3]);
end

code_layer_transfer = imresize(code_layer,[Imrow Imcol]);
% code_layer = code_layer(:);
% code_layer = (code_layer-min(code_layer))/(max(code_layer)-min(code_layer))*255;
% cae_features_rgb2 = uint8(reshape(code_layer,Imrow,Imcol,[]));

% dimension reduction

NewChannels = cat(3,TestIm, code_layer_transfer);
%   NewChannels = code_layer_transfer;

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

%% seg using pca merge to 3

%parameter evaluation
nParameter = 11;
KernelSize = 2:4:42;
parameter1 = repmat(KernelSize,1,nParameter);
parameter2 = repmat(KernelSize',1,nParameter)';
parameter2 = parameter2(:)';
KernelParamater = [parameter1;parameter2];

parfor k= 1:nParameter^2
[labels,modes,regSize] = edison_wrapper(DimReductionIm , @RGB2Luv,...
'SpatialBandWidth',KernelParamater(1,k),...
'RangeBandWidth',KernelParamater(2,k));


%% imresize
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
segs{k} = reshape(NewPixelArray,SegrowNew,SegcolNew);

end

save(['/net/linse8/no_backup_01/s1184/edition_test/test-bsds500/',...
Imname],'segs');

end;
toc;