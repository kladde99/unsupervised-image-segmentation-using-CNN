% input RGB and features, reduce the dimension to 2 and then plot it
clc;
clear all;
close all;
%% load the data and matcovnet toolbox
load('../quellcode/Training_BSDS500/data/test/net-cae1.mat');
load('../quellcode/Training_horse/imdb.mat');
load('../quellcode/Training_horse/groundTruth_binary.mat');
run('../quellcode/matconvnet-new-solvers/matlab/vl_setupnn');
k=9;% the index of the code layer

net.layers(end) = [];

for i  = 1:10 % index of the image to analyze

im = imdb.images.data(:,:,:,i);
[Imrow ,Imcol,depth] = size(im);
res = vl_simplenn(net,im) ;
code_layer = res(k).x;

code_layer_transfer = imresize(code_layer,[Imrow Imcol]);
% the dimension reduce to 2 
DimReductionIm = horse_DimReductionPCA([],im,2,1);
DimReductionFeature = horse_DimReductionPCA(code_layer_transfer,im,2,0);
DimReductionFeatureRGB = horse_DimReductionPCA(code_layer_transfer,im,2,1);

% norm
DimReductionIm_norm= featureNorm(DimReductionIm);
DimReductionFeature_norm= featureNorm(DimReductionFeature );
DimReductionFeatureRGB_norm = featureNorm( DimReductionFeatureRGB);
% plot the visualization of PCA-transformed features                                 
dim1 = DimReductionIm_norm(:,:,1);
x=dim1(:);
dim2 = DimReductionIm_norm(:,:,2);
y=dim2(:);

Dim1 = DimReductionFeature_norm(:,:,1);
X=Dim1(:);
Dim2 = DimReductionFeature_norm(:,:,2);
Y=Dim2(:);

dimx1 = DimReductionFeatureRGB_norm(:,:,1);
x1=dimx1(:);
dimx2 = DimReductionFeatureRGB_norm(:,:,2);
y1=dimx2(:);

gtValue = groundTruth{i};
gtValue_vector = gtValue(:);

% let the color of value 0 (pixels of background)to be red
c= zeros(numel(gtValue_vector),3);
c1 = double(gtValue_vector);
idx_blue = find(c1==0);idx_red = find(c1==1);
c(idx_blue,3)=  1;c(idx_red,1)=1;

C= zeros(numel(gtValue_vector),3);
C1 = double(gtValue_vector);
idx_Blue = find(C1==0);idx_Red = find(C1==1);
C(idx_Blue,3)=  1;C(idx_Red,1)=1;

c_x1= zeros(numel(gtValue_vector),3);
c_1 = double(gtValue_vector);
idx_blue1 = find(c_1==0);idx_red1 = find(c_1==1);
c_x1(idx_blue1,3)=  1;c_x1(idx_red1,1)=1;

% plot different channels
figure(10*i);
subplot(3,2,1);imagesc(DimReductionIm_norm(:,:,1));axis image off ;colormap gray; 
title('RGB Channel1');
subplot(3,2,2);imagesc(DimReductionIm_norm(:,:,2));axis image off ;colormap gray; 
title('RGB Channel2');
subplot(3,2,3);imagesc(DimReductionFeature_norm(:,:,1));axis image off ;colormap gray; 
title('Feature Channel1');
subplot(3,2,4);imagesc(DimReductionFeature_norm(:,:,2));axis image off ;colormap gray; 
title('Feature Channel2');
subplot(3,2,5);imagesc(DimReductionFeatureRGB_norm(:,:,1));axis image off ;
title('features RGB');
subplot(3,2,6);imagesc(DimReductionFeatureRGB_norm(:,:,2));axis image off ;
title('features RGB');
% print the result
myfilename=['id_',num2str(i),'_HorseFeatureMaps_horsenet','.png'];
saveas(gcf,myfilename); 
% plot features
figure(i);
ax_rgb = subplot(1,2,1);
s1 = scatter3(ax_rgb,x,y,gtValue_vector,2,c);  view(2);
s1.Marker = '.';
s1.DisplayName = 'horse';legend('show');
title('Original Image');

ax_feature = subplot(1,2,2);
s2 = scatter3(ax_feature,X,Y,gtValue_vector,2,C);  view(2);
s2.Marker = '.';
s2.DisplayName = 'horse';legend('show');
title('Learned Features');
% print the result
ax_featureRGB = subplot(1,3,3);
s3 = scatter3(ax_featureRGB,x1,y1,gtValue_vector,2,c_x1);  view(2);
 s3.Marker = '.';
 s3.DisplayName = 'horse';legend('show');
 title('learned Features and RGB fusion');
 myfilename=['id_',num2str(i),'_HorseFeatureAnalysis','.jpg'];
 saveas(gcf,myfilename); 
end