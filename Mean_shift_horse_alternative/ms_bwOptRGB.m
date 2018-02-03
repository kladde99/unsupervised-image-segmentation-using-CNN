%% mean shift + channels normalization + remove zero matrices+ 16channels
%% + 2 group of features combination+ spatial information+ parameter optimization for RGB
%% using grid search
close all;
clear all;
clc;

run('/Users/Ben/Documents/MATLAB/matconvnet-1.0-beta20/matlab/vl_setupnn');
load('/Users/Ben/Documents/MATLAB/horse/data/test-samenumber/net-epoch-2000.mat');

load('/Users/Ben/Documents/MATLAB/horse/imdb.mat');

net.layers(end) = [] ;

%% spatial information feature
rowid = [];
colid = []; colelement = (1:144)';
for i = 1: 160
     
     rowid = [rowid;i*ones(144,1)];
     colid = [colid;colelement];
end
%normalization
rowid = (rowid - min(rowid))/(max(rowid)-min(rowid));
colid = (colid -min(colid))/(max(colid)-min(rowid));

for imidx=20:2:34;
res = vl_simplenn(net, imdb.images.data(:,:,:,imidx)) ;
 %% print the  segmentation figures
figure(imidx) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(4,3,1) ;
imagesc(res(1).x) ; axis image off  ;
title('original image') ; hold on;   
%% only RGB
for bandwidth   = 0.1 :0.1:1
featureRGB = reshape(res(1).x,[],3);
feature_matrix_rgb1 = [featureRGB,rowid,colid];
[imSeg_rgb nClustRGB] = Ms2(feature_matrix_rgb1,res(1).x,bandwidth);

%% print the  segmentation figures

set(gcf,'name', 'Part 2: network output') ;
subplot(4,3,bandwidth*10+1) ;
imagesc(imSeg_rgb) ; axis image off  ;
title(['rgbBW=',num2str(bandwidth) ' ','Num of clust=',num2str(nClustRGB)]) ;hold on;

end

myFileName=['im' num2str(imidx),'rgb' '.jpg'];
saveas(gcf,myFileName);
end