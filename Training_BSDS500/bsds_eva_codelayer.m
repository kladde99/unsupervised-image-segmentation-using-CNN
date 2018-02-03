%% show the feature maps of the code layer
close all;
clear all;
clc;
%% load images, toolbox and network we have trained
run('../quellcode/matconvnet-new-solvers/matlab/vl_setupnn');
load('../quellcode/Training_BSDS500/data/test/net-cae1.mat');
load('../quellcode/Training_BSDS500/imdb_test.mat');
% index of the code layer
k = 9;
net.layers(k:end) = [] ;

%%  code layer
for a = [1 2 5 9] % index of the images
res = vl_simplenn(net, im2single(imdb3.images {1,a})) ;

%% original image
figure(a) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(2,5,1) ;
imagesc(res(1).x) ; axis image off  ;
title('Original Image') ;

for i= 1:8 % num of the channels of the code layer
    im =imresize( res(end).x,[128 192]);
    im2= (im(:,:,i));
    subplot(2,5,i+1) ;
    imagesc((im2)) ; colormap gray; axis image off  ;
    title(['No.',num2str(i),' Channel']) ;
    
end
% print the results
myFileName=['t4code layer','_im', num2str(a) '.pdf'] ;
saveas(gcf,myFileName);
end