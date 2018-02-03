close all;
clear all;
clc;
% addpath(genpath('/net/linse8/no_backup_01/s1184'));

cd /home/s1184/matlab/edition_test/test2segfeatureavg/;
run('/home/s1184/matlab/matconvnet-new-solvers/matlab/vl_setupnn');
load('/home/s1184/matlab/bsds500works/test2/net-epoch-900.mat');
load('/home/s1184/matlab/edition_test/imdb_test.mat');
load('/home/s1184/matlab/edition_test/TestImageIdx.mat');

net.layers(end) = [];
segs = cell(0);

for id= 1:10

Imname =[files(id).name(1:end-4) '.mat'];
TestIm = im2single(imdb3.images{id});
[Imrow,Imcol,Imdepth] = size(TestIm);

% SmoothIm = uint8(vl_imsmooth(TestIm,2));
sign = 0;
if Imrow > Imcol
TestIm = permute(TestIm,[2 1 3]);
sign = 1;
end
%    TestIm = imresize(TestIm,[128 192]);
res = vl_simplenn(net, TestIm) ;

im_code = res(9).x;

if sign ==1
 im_code = permute(im_code,[2 1 3]);
 TestIm = permute(TestIm,[2 1 3]);
end

figure(id) ; clf  ;  % if output color image
set(gcf,'name', 'Part 1: network input') ;
subplot(1,2,1) ;
imagesc(res(1).x) ; axis image off  ;
title('original image') ;


im_avg = sum(im_code,3)/size(im_code,3);
im_avg = (imresize(im_avg,[Imrow Imcol]));
subplot(1,2,2);imagesc(im_avg);title([num2str(id),' imAVG']) ;
colormap gray;  axis image off  ;

myfilename=['id_',num2str(id),'_featureavg','.jpg'];
saveas(gcf,myfilename);

SpatialBW  = 32; RangeBW  = 22;

S = msseg(im_avg,SpatialBW,RangeBW)  ;                         

%%imresize
SegIm =(uint16(im2uint8(S)));
[Segrow Segcol]= size(SegIm);

if Segrow > Segcol
SegIm_new = imresize(SegIm,[481 321]);
else
SegIm_new = imresize(SegIm,[321 481]);
end
[SegrowNew SegcolNew]= size(SegIm_new);

OldPixelArray = SegIm_new(:);
NewPixelArray = OldPixelArray;
SortOldPixelArray=sort(unique(OldPixelArray));
ncluster=length(SortOldPixelArray);

for i=1:ncluster
idx=find(OldPixelArray==SortOldPixelArray(i));
NewPixelArray(idx)=i;
end
segs = reshape(NewPixelArray,SegrowNew,SegcolNew);

%% plot
figure(10*id);
subplot(1,2,1);
imagesc(imdb3.images{id});axis image off;
title('original image');
subplot(1,2,2);
imagesc(label2rgb(segs));axis image off;colormap gray;
title('seg using featuresAVG');
myfilename=['id_',num2str(id),'_segUsingfeaturesavg','.jpg'];
saveas(gcf,myfilename);

end;