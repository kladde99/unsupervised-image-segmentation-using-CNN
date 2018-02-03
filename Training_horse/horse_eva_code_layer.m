%% show the feature maps of the code layer from the network trained on Weizmann horse dataset
close all;
clear all;
clc;
%% initialize the matconv toolbox and image dataset
run('../quellcode/matconvnet-new-solvers/matlab/vl_setupnn');
% it depends on different net-epoch
load('../quellcode/Training_horse/data/test/net-epoch-1000.mat');
load('../quellcode/Training_horse/imdb.mat');

% the layer index of the code layer
for k = 7 
    
net.layers(k:end) = [] ;

%  code layer
for a = 1:10 % index of the image
res = vl_simplenn(net, imdb.images.data(:,:,:,a)) ;

%% original image
%  figure(1) ; clf ; colormap gray ;
figure(a) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(4,5,1) ;
imagesc(res(1).x) ; axis image off  ;
title('original image') ;


for i= 1:16 % number of the channels of the code layer
    im =imresize( res(end).x,[144 160]);
    im2= (im(:,:,i));
    subplot(4,5,i+1) ;
    imagesc((im2)) ; colormap gray; axis image off  ;
    title(['No.',num2str(i),' Channel']) ;
    
end
% print the result
myFileName=['layer_',num2str(k),'_im', num2str(a) '.eps'] ;
saveas(gcf,myFileName);
end

end
 