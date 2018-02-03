%% show the output of different layers from CAE 1 or 2 from BSDS 500
close all;
clear all;
clc;

%% load the data and toolbox
run('../quellcode/matconvnet-new-solvers/matlab/vl_setupnn');
load('../quellcode/Training_BSDS500/data/test/net-cae1.mat');
load('../quellcode/Training_BSDS500/imdb_test.mat');

for a = [2 5 9] % index of the test image
 for  k = 3  % the index of conv layer you want to show
net.layers(k:end) = [] ;

%% the corresponding k layer
res = vl_simplenn(net, im2single(imdb3.images {1,a})) ;

%% original image
figure(k) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(2,5,1) ;
imagesc(res(1).x) ; axis image off  ;
title('Original Image') ;


for i= 1:9 % num of the channels you want to display
    im =imresize( res(end).x,[128 192]);
    im2= (im(:,:,i));
    subplot(2,5,i+1) ;
    imagesc((im2)) ; colormap gray; axis image off  ;
    title(['No.',num2str(i),' Channel']) ;
    
end
% print the result
myFileName=['t4layer',num2str(k),'_im', num2str(a) '.pdf'] ;
saveas(gcf,myFileName);
 end
end
