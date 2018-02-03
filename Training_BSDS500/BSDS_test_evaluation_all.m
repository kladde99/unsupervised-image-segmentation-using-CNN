%% show the output of different layers
close all;
clear all;
clc;
%% load images, toolbox and network we have trained
run('../quellcode/matconvnet-new-solvers/matlab/vl_setupnn');
load('../quellcode/Training_BSDS500/data/test/net-cae1.mat');
load('../quellcode/Training_BSDS500/imdb_test.mat');
% load('/Users/Ben/Documents/MATLAB/BSDS500/data/test-reluless2/net-epoch-6401.mat');

net.layers(end) = [] ;

for a = 1:10
TestIm = single(imdb3.images{a});
[Imrow,Imcol,Imdepth] = size(TestIm);

sign = 0;
if Imrow > Imcol
   TestIm = permute(TestIm,[2 1 3]);
   sign = 1;
end
%    TestIm = imresize(TestIm,[128 192]);
res = vl_simplenn(net, TestIm) ;
%% original image
%  figure(1) ; clf ; colormap gray ;
figure(a) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(6,6,1) ;
imagesc(uint8(res(1).x)) ; axis image off  ;
title('CNN input') ;

%% reconstruction

% im= reshape(res(end).x,128,192,1,[]);

for i = 1
    
    subplot(6,6,i+1) ;
    imagesc(uint8(res(end).x)) ; axis image off  ;
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
TestIm = single(imdb3.images{a});
[Imrow,Imcol,Imdepth] = size(TestIm);

sign = 0;
if Imrow > Imcol
   TestIm = permute(TestIm,[2 1 3]);
   sign = 1;
end
%    TestIm = imresize(TestIm,[128 192]);
res = vl_simplenn(net, TestIm) ;
%% original image
%  figure(1) ; clf ; colormap gray ;
figure(a) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(6,6,1) ;
imagesc(uint8(res(1).x)) ; axis image off  ;
title('CNN input') ;

%% layer14
im= reshape(res(end).x,8,12,1,[]);

for i = 1:4
    
    subplot(6,6,i+1) ;
    imagesc(im(:,:,:,i)) ; axis image off  ;
    title('L14') ;
    
end
%    set(gcf,'PaperUnits','inches','PaperPosition',[0 0 36 20])
 myFileName=['12_8_' num2str(a) '.jpg'] ;
%  print('-djpeg',myFileName,'-r100');
saveas(gcf,myFileName);
end

%% layer 12

% Deploy: remove loss
 net.layers(12:end) = [] ;

 for a = 1:10
     
TestIm = single(imdb3.images{a});
[Imrow,Imcol,Imdepth] = size(TestIm);

sign = 0;
if Imrow > Imcol
   TestIm = permute(TestIm,[2 1 3]);
   sign = 1;
end
%    TestIm = imresize(TestIm,[128 192]);
res = vl_simplenn(net, TestIm) ;

%  figure(1) ; clf ; colormap gray ;
figure(a) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(6,6,1) ;
imagesc(uint8(res(1).x)) ; axis image off  ;
title('CNN input') ;

im= reshape(res(end).x,16,24,1,[]);

for i = 1:16
    
    subplot(6,6,i+1) ;
    imagesc(im(:,:,:,i)) ; axis image off  ;
    title('L12') ;
    
end
% set(gcf,'PaperUnits','inches','PaperPosition',[0 0 36 20])
 myFileName=['24_16_' num2str(a) '.jpg'] ;
%  print('-djpeg',myFileName,'-r100');
saveas(gcf,myFileName);
 end
 
 
 %% layer 09
 
 % Deploy: remove loss
 net.layers(9:end) = [] ;

 for a = 1:10

 TestIm = single(imdb3.images{a});
[Imrow,Imcol,Imdepth] = size(TestIm);

sign = 0;
if Imrow > Imcol
   TestIm = permute(TestIm,[2 1 3]);
   sign = 1;
end
%    TestIm = imresize(TestIm,[128 192]);
res = vl_simplenn(net, TestIm) ;
%  figure(1) ; clf ; colormap gray ;

figure(a) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(6,6,1) ;
imagesc(uint8(res(1).x)) ; axis image off  ;
title('CNN input') ;

im= reshape(res(end).x,32,48,1,[]);

for i = 1:16
    
    subplot(6,6,i+1) ;
    imagesc(im(:,:,:,i)) ; axis image off  ;
    title('L09') ;
    
end
    
 myFileName=['48_32_' num2str(a) '.jpg'] ;
saveas(gcf,myFileName);
 end
 
 
 %% layer 06
 % Deploy: remove loss
 net.layers(6:end) = [] ;

 for a = 1:10
     
TestIm = single(imdb3.images{a});
[Imrow,Imcol,Imdepth] = size(TestIm);

sign = 0;
if Imrow > Imcol
   TestIm = permute(TestIm,[2 1 3]);
   sign = 1;
end
%    TestIm = imresize(TestIm,[128 192]);
res = vl_simplenn(net, TestIm) ;
%  figure(1) ; clf ; colormap gray 
figure(a) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(6,6,1) ;
imagesc(uint8(res(1).x)) ; axis image off  ;
title('CNN input') ;

im= reshape(res(end).x,64,96,1,[]);

for i = 1:16
    
    subplot(6,6,i+1) ;
    imagesc(im(:,:,:,i)) ; axis image off  ;
    title('L06') ;
    
end

 myFileName=['96_64_' num2str(a) '.jpg'] ;
 saveas(gcf,myFileName);
 end

