close all;
clear all;
clc;

load('../quellcode/Training_BSDS500/imdb_test.mat');
load('../quellcode/Training_BSDS500/TestImageIdx.mat');
segs = cell(0);

for id= 1:10

Imname =[files(id).name(1:end-4) '.mat'];
TestIm = im2single(imdb3.images{id});
[Imrow,Imcol,Imdepth] = size(TestIm);


SpatialBW  = 20; RangeBW  = 16;

[labels,modes,regSize] = edison_wrapper(imdb3.images{id}, @RGB2Luv,...
'SpatialBandWidth',SpatialBW,...
'RangeBandWidth',RangeBW);

%%imresize
SegIm =rgb2gray (uint16(labels));
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
figure(id);
subplot(1,2,1);
imagesc(imdb3.images{id});axis image off;
title('original image');
subplot(1,2,2);
imagesc(label2rgb(segs));axis image off;colormap gray;
title('seg using original Im');
myfilename=['id_',num2str(id),'_segUsingOriginIm','.jpg'];
saveas(gcf,myfilename);

end;