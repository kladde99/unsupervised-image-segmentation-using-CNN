close all;
clear all;
clc;
% addpath(genpath('/net/linse8/no_backup_01/s1184'));

cd /home/s1184/matlab/edition_test/test4segpcamerge1/;
run('/home/s1184/matlab/matconvnet-new-solvers/matlab/vl_setupnn');
load('/home/s1184/matlab/bsds500works/test4/net-epoch-900.mat');
load('/home/s1184/matlab/edition_test/imdb_test.mat');
load('/home/s1184/matlab/edition_test/TestImageIdx.mat');
k=6;

net.layers(end) = [];
% segs = cell(0);

for id= 1:10

% Imname =[files(id).name(1:end-4) '.mat'];
TestIm = im2single(imdb3.images{id});
[Imrow,Imcol,Imdepth] = size(TestIm);

sign = 0;
if Imrow > Imcol
TestIm = permute(TestIm,[2 1 3]);
sign = 1;
end

res = vl_simplenn(net, TestIm) ;

code_layer = res(k).x;

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

DimReductionIm  = TransferIm_new (:,:,1);

figure(id) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(1,2,1) ;
imagesc(TestIm) ; axis image off  ;
title('original image') ;

subplot(1,2,2);
imagesc(DimReductionIm);title([num2str(id),' pca to 1']) ;
colormap gray;  axis image off  ;

myfilename=['id_',num2str(id),'_featurePCAto1','.jpg'];
saveas(gcf,myfilename);

%% seg using pca merge to one

SpatialBW  = 22; RangeBW  = 18;

S = msseg(DimReductionIm,SpatialBW,RangeBW)  ;                         
v= S(:);
n= (v-min(v))/(max(v)-min(v));
S_norm= (reshape(n,[Imrow Imcol]));

%%imresize
SegIm =(uint16(im2uint8(S_norm)));
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

%% seg using original image
SpatialBW  = 20; RangeBW  = 16;

[labels,modes,regSize] = edison_wrapper(imdb3.images{id}, @RGB2Luv,...
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
segs_origin = reshape(NewPixelArray,SegrowNew,SegcolNew);


%% plot
figure(10*id);
subplot(1,3,1);
imagesc(imdb3.images{id});axis image off;
title('original image');
subplot(1,3,2);
imagesc(label2rgb(segs));axis image off;colormap gray;
title('seg pca to 1');
subplot(1,3,3);
imagesc(label2rgb(segs_origin));axis image off;colormap gray;
title('seg origin');
myfilename=['id_',num2str(id),'_segUsingfeaturesPCAto1','.jpg'];
saveas(gcf,myfilename);

end;