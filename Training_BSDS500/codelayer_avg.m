close all;
clear all;
clc;

run('../quellcode/matconvnet-new-solvers/matlab/vl_setupnn');
load('../quellcode/Training_BSDS500/data/test/net-cae1.mat');
load('../quellcode/Training_BSDS500/imdb_test.mat');

k=9;
net.layers(k:end) = [] ;
%%  code layer
for a = 1:10
res = vl_simplenn(net, imdb.images.data(:,:,:,a)) ;

%% original image
figure(a) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(1,2,1) ;
imagesc(res(1).x) ; axis image off  ;
title('original image') ;

im_code = res(end).x;
im_avg = sum(im_code,3)/size(im_code,3);
im_avg = imresize(im_avg,[192 128]);
subplot(1,2,2);imagesc(im_avg);title([num2str(a),' imAVG']) ;
colormap gray;  axis image off  ;
myFileName=['avgIm_',num2str(a) '.jpg'] ;
saveas(gcf,myFileName);

end