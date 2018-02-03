% pretrain model feature map analysis

close all;
clear all;
clc;
%% load the data, toolbox and imagenet model
run('../quellcode/matconvnet-new-solvers/matlab/vl_setupnn');
load('../quellcode/Training_BSDS500/imdb_test.mat');
load('../quellcode/Training_BSDS500/TestImageIdx.mat');% index of the test images
net = load('../quellcode/Training_horse/imagenet-vgg-f.mat') ;
net = vl_simplenn_tidy(net) ;

for k=[13 11 7 3] % index the corresponding conv layer

for id= 1:10  % index of the images
 
TestIm = im2single(imdb3.images{id});
[Imrow,Imcol,Imdepth] = size(TestIm);

im = imresize(im2single(imdb3.images{id}),...
net.meta.normalization.imageSize(1:2)) ;
res = vl_simplenn(net, im) ;

im_code = res(k).x;

%% feature averaging
im_avg = sum(im_code,3)/size(im_code,3);
im_avg = (imresize(im_avg,[Imrow Imcol]));

% norm before transfer to uint8 format 
v = im_avg(:);
n= (v-min(v))/(max(v)-min(v));
im_avg_uint = im2uint8(reshape(n,[Imrow Imcol]));

%% pca merge to one
code_layer_transfer = imresize(im_code,[Imrow Imcol]);
NewChannels = code_layer_transfer;
%  normalization
 NewChannels_vector = NewChannels(:);
 NewChannels_norm = (NewChannels_vector-min(NewChannels_vector))/...
                 (max(NewChannels_vector)-min(NewChannels_vector));

FeatureVector = reshape(NewChannels_norm,...
    size(NewChannels,1)*size(NewChannels,2),size(NewChannels,3));
[coeff,score,latent] = pca(FeatureVector);
TransferIm = FeatureVector*coeff;
             
TransferIm_new = im2uint8(reshape(TransferIm,size(NewChannels,1),...
    size(NewChannels,2),size(NewChannels,3)));
         
 DimReductionIm  = TransferIm_new (:,:,1);
 
 %% plot
figure(id) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(1,3,1) ;
imagesc(TestIm) ; axis image off  ;
title('original image') ;

subplot(1,3,2);imagesc(im_avg);title([num2str(id),' imAVG']) ;
colormap gray;  axis image off  ;

subplot(1,3,3);
imagesc(DimReductionIm);title([num2str(id),' pca to 1']) ;
colormap gray;  axis image off  ;

myfilename=['layer_',num2str(k),'_id_',num2str(id),...
    '_preTrainFeatureShow','.jpg'];
saveas(gcf,myfilename);

%% seg using average feature map

SpatialBW  = 22; RangeBW  = 18;

S = msseg(im_avg_uint,SpatialBW,RangeBW)  ;                         
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
segs2 = reshape(NewPixelArray,SegrowNew,SegcolNew);


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
 subplot(1,4,1);
imagesc(imdb3.images{id});axis image off;
title('original image');
 subplot(1,4,2);
 imagesc(label2rgb(segs));axis image off;colormap gray;
 title('seg featuresAVG');
 subplot(1,4,3);
 imagesc(label2rgb(segs2));axis image off;colormap gray;
 title('seg pca to 1');
 subplot(1,4,4);
 imagesc(label2rgb(segs_origin));axis image off;colormap gray;
 title('seg origin');
 myfilename=['layer_',num2str(k),'_id_',num2str(id),...
     '_pretrainFeatureAnalysis','.jpg'];
saveas(gcf,myfilename);
 
end;

end;