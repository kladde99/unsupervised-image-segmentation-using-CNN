%% show the output of image reconstruction
close all;
clear all;
clc;

%% load images, toolbox and network we have trained
run('../quellcode/matconvnet-new-solvers/matlab/vl_setupnn');
load('../quellcode/Training_BSDS500/data/test/net-cae1.mat');
load('../quellcode/Training_BSDS500/imdb_test.mat');
net.layers(end) = [] ;

 %% print
for a = [1 2 5 9] % index of the image
    im = imdb3.images{1,a};
    im_blur= vl_imsmooth(im2single(im),3);
res = vl_simplenn(net,im_blur ) ;

%% original image
figure(a) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(1,2,1) ;
imagesc(res(1).x) ; axis image off  ;
title('Original Image') ;

%% reconstruction 
    subplot(1,2,2) ;
    imagesc((res(end).x)) ; axis image off  ;
    title('Image Reconstruction') ;
    % print the results
    myFileName=['recon_' num2str(a) '.png'] ;
    saveas(gcf,myFileName);
end

