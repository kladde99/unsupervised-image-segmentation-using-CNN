% display the feature maps of different conv layers of imagenet pretrained
% model
close all;
clear all;
clc;

%% load data , toolbox and pretrained model
run('../quellcode/matconvnet-new-solvers/matlab/vl_setupnn');
load('../quellcode/Training_horse/imdb.mat');

net = load('../quellcode/Training_horse/imagenet-vgg-f.mat') ;
net = vl_simplenn_tidy(net) ;

for i =[2 4 6 8 10] % index of images
    
im = imresize(imdb.images.data(:,:,:,i), net.meta.normalization.imageSize(1:2)) ;

res = vl_simplenn(net, im) ;

for id = [15 13 11 7 3] % index of the conv layer

figure(id-1);clf;
subplot(7,10,1); imagesc(im);axis image off;
for a = 1 : 64
    
    im2 = imresize(res(id).x(:,:,a),[224 224]);
    im3 = im2uint8(im2);
    subplot(7,10,a+1); imagesc((im3));colormap gray; axis image off;
    
end
% print the result
myFileName=['id_',num2str(i),'_featureMap', num2str(id) ,'.jpg'] ;
saveas(gcf,myFileName);

end
 
end
 