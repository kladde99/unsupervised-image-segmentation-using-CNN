%% show the effect of image reconstruction on Weizmann horse dataset
close all;
clear all;
clc;
%% initialize the matconv toolbox and image dataset
run('../quellcode/matconvnet-new-solvers/matlab/vl_setupnn');
% it depends on different net-epoch
load('../quellcode/Training_horse/data/test/net-epoch-1000.mat');
load('../quellcode/Training_horse/imdb.mat');

net.layers(end) = [] ;

 %% loop 10 image for test
for a = 1:10
res = vl_simplenn(net, imdb.images.data(:,:,:,a)) ;

%% original image
%  figure(1) ; clf ; colormap gray ;
figure(a) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(1,2,1) ;
imagesc(res(1).x) ; axis image off  ;
title('CNN input') ;

%% reconstruction
    subplot(1,2,2) ;
    imagesc((res(end).x)) ; axis image off  ;
    title('recon') ;
    
 % print the result
 myFileName=['re_con_' num2str(a) '.jpg'] ;
saveas(gcf,myFileName);
end

