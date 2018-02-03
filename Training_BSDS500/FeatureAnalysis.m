close all;
clear all;
clc;
%% load data and toolbox
run('../quellcode/matconvnet-new-solvers/matlab/vl_setupnn');
load('../quellcode/Training_BSDS500/data/test/net-cae1.mat');
load('../quellcode/Training_BSDS500/imdb_test.mat');
load('../quellcode/Training_BSDS500/TestImageIdx.mat');

k=9;% id of the code layer

net.layers(end) = [];

for id= [1 2 5 9] % index of the images 
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
% averaginf
 im_avg = sum(code_layer,3)/size(code_layer,3);
 im_avg = (imresize(im_avg,[Imrow Imcol]));
 % norm before transfer to uint8 format 
 v = im_avg(:);
 n= (v-min(v))/(max(v)-min(v));
 im_avg_uint = im2uint8(reshape(n,[Imrow Imcol]));

  DimReductionIm_fusion3 = DimReductionPCA(code_layer_transfer,TestIm,3,0);
  DimReductionIm_fusion1 = DimReductionPCA(code_layer_transfer,TestIm,1,0);
 DimReductionIm_Imfusion3 = DimReductionPCA(code_layer_transfer,TestIm,3,1);
 DimReductionIm_Imfusion1 = DimReductionPCA(code_layer_transfer,TestIm,1,1);

figure(id) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(2,5,1) ;
imagesc(TestIm) ; axis image off  ;
title('Original Image') ;

 subplot(2,5,2);
 imagesc(DimReductionIm_fusion3);title('pca 3') ;
   axis image off  ;

  subplot(2,5,3);
 imagesc(DimReductionIm_fusion1);title('pca 1') ;
  colormap gray; axis image off  
 
 subplot(2,5,4);
imagesc(DimReductionIm_fusion3(:,:,1));title('No.1 Channel') ;
 colormap gray; axis image off  ;
 
 subplot(2,5,5);
imagesc(DimReductionIm_fusion3(:,:,2));title('No.2 Channel') ;
 colormap gray; axis image off  ;

 subplot(2,5,6);
imagesc(DimReductionIm_fusion3(:,:,3));title('No.3 Channel') ;
 colormap gray; axis image off  ;
  
 subplot(2,5,7);
imagesc(DimReductionIm_fusion1);title('1 Channel') ;
  colormap gray; axis image off  ;
 subplot(2,5,8);
 imagesc(im_avg_uint);title('featuremap avg') ;
 colormap gray; axis image off ;
  % print the result
myfilename=['id_',num2str(id),'_pca_withoutImt4','.png'];
saveas(gcf,myfilename);

end;