%% show the output of different layers
close all;
clear all;
clc;
%% initialize the matconv toolbox and image dataset
run('/Users/Ben/Documents/MATLAB/matconvnet-new-solvers/matlab/vl_setupnn');
% it depends on different net-epoch
load('/Users/Ben/Documents/MATLAB/horse2_works/data/test/net-epoch-1000.mat');
load('/Users/Ben/Documents/MATLAB/horse2_works/imdb.mat');

net.layers(end) = [] ;

 %% print
for a = 1:10
res = vl_simplenn(net, imdb.images.data(:,:,:,a)) ;

%% original image
%  figure(1) ; clf ; colormap gray ;
figure(a) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(6,6,1) ;
imagesc(res(1).x) ; axis image off  ;
title('CNN input') ;

%% reconstruction

% im= reshape(res(end).x,128,192,1,[]);

for i = 1
    
    subplot(6,6,i+1) ;
    imagesc((res(end).x)) ; axis image off  ;
    title('recon') ;
    
end
%    set(gcf,'PaperUnits','inches','PaperPosition',[0 0 36 20])
 myFileName=['re_con_' num2str(a) '.jpg'] ;
%  print('-djpeg',myFileName,'-r100');
saveas(gcf,myFileName);
end



%% Deploy: remove loss
 net.layers(14:end) = [] ;

 %% print
for a = 1:10
res = vl_simplenn(net, imdb.images.data(:,:,:,a)) ;

%% original image
%  figure(1) ; clf ; colormap gray ;
figure(a) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(6,6,1) ;
imagesc(res(1).x) ; axis image off  ;
title('CNN input') ;

%% layer14
im= reshape(res(end).x,9,10,1,[]);

for i = 1:2
    
    subplot(6,6,i+1) ;
    imagesc(im(:,:,:,i)) ; axis image off  ;
    title('9*10&12filters') ;
    
end
   
 myFileName=['9_10_' num2str(a) '.jpg'] ;
 saveas(gcf,myFileName);
end

%% layer 12

% Deploy: remove loss
 net.layers(12:end) = [] ;

 for a = 1:10
     
res = vl_simplenn(net, imdb.images.data(:,:,:,a)) ;
%  figure(1) ; clf ; colormap gray ;
figure(a) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(6,6,1) ;
imagesc(res(1).x) ; axis image off  ;
title('CNN input') ;

im= reshape(res(end).x,18,20,1,[]);

for i = 1:16
    
    subplot(6,6,i+1) ;
    imagesc(im(:,:,:,i)) ; axis image off  ;
    title('18*20&384filters') ;
    
end
% set(gcf,'PaperUnits','inches','PaperPosition',[0 0 36 20])
 myFileName=['18_20_' num2str(a) '.jpg'] ;
%  print('-djpeg',myFileName,'-r100');
saveas(gcf,myFileName);
 end
 
 
 %% layer 09
 
 % Deploy: remove loss
 net.layers(9:end) = [] ;

 for a = 1:10
res = vl_simplenn(net, imdb.images.data(:,:,:,a)) ;
%  figure(1) ; clf ; colormap gray ;

figure(a) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(6,6,1) ;
imagesc(res(1).x) ; axis image off  ;
title('CNN input') ;

im= reshape(res(end).x,36,40,1,[]);

for i = 1:16
    
    subplot(6,6,i+1) ;
    imagesc(im(:,:,:,i)) ; axis image off  ;
    title('36*40&192filters') ;
    
end
%     set(gcf,'PaperUnits','inches','PaperPosition',[0 0 36 20])
 myFileName=['36_40_' num2str(a) '.jpg'] ;
%  print('-djpeg',myFileName,'-r100');
saveas(gcf,myFileName);
 end
 
 
 %% layer 06
 % Deploy: remove loss
 net.layers(6:end) = [] ;

 for a = 1:10
     
res = vl_simplenn(net, imdb.images.data(:,:,:,a)) ;
%  figure(1) ; clf ; colormap gray 
figure(a) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(6,6,1) ;
imagesc(res(1).x) ; axis image off  ;
title('CNN input') ;

im= reshape(res(end).x,72,80,1,[]);

for i = 1:16
    
    subplot(6,6,i+1) ;
    imagesc(im(:,:,:,i)) ; axis image off  ;
    title('72*80&96filters') ;
    
end
% print the result
 myFileName=['72_80_' num2str(a) '.jpg'] ;
saveas(gcf,myFileName);
 end

