% EDISON of the original image
close all;
clear all;
clc;

load('../quellcode/Training_BSDS500/imdb_test.mat');
load('../quellcode/Training_BSDS500/TestImageIdx.mat');

segs = cell(0);
tic;
for id=1:200

Imname =[files(id).name(1:end-4) '.mat'];
TestIm = imdb3.images{id};
[Imrow,Imcol,Imdepth] = size(TestIm);

%% parameter evaluation
nParameter = 15;
KernelSize = 2:5:72;
parameter1 = repmat(KernelSize,1,nParameter);
parameter2 = repmat(KernelSize',1,nParameter)';
parameter2 = parameter2(:)';
KernelParamater = [parameter1;parameter2];

for k= 1:nParameter^2

[labels,modes,regSize] = edison_wrapper(TestIm , @RGB2Luv,...
'SpatialBandWidth',KernelParamater(1,k),...
'RangeBandWidth',KernelParamater(2,k));

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
segs{k} = reshape(NewPixelArray,SegrowNew,SegcolNew);


end;
save(['/net/linse8/no_backup_01/s1184/edition_test/test-bsds500rgb/',Imname],'segs');

end;

toc;