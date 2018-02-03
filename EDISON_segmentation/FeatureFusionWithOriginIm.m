% show the feature fusion of the  original image and features learned from
% CNN

close all;
clear all;
clc;
%% load the data and toolbox
run('/Users/Ben/Documents/MATLAB/matconvnet-new-solvers/matlab/vl_setupnn');
load('/Users/Ben/Documents/MATLAB/bsds500works/data/test5/net-epoch-1000.mat');
load('/Users/Ben/Documents/MATLAB/bsds500works/imdb_test.mat');
load('/Users/Ben/Documents/MATLAB/bsds500works/TestImageIdx.mat');

k=9;

for id= 1:10

TestIm = im2single(imdb3.images{id});
[Imrow,Imcol,Imdepth] = size(TestIm);

sign = 0;
if Imrow < Imcol
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
% dimension reduction

NewChannels = cat(3,TestIm, code_layer_transfer);
% uncomment to get features without adding RGB features
% NewChannels = code_layer_transfer; 

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
imagesc(DimReductionIm);title([num2str(id),' Feature+Im to 1']) ;
colormap gray;  axis image off  ;

myfilename=['id_',num2str(id),'_featurePCAto1','.jpg'];
saveas(gcf,myfilename);

end