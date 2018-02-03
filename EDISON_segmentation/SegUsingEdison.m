function [segs,ncluster]= SegUsingEdison(TestIm,DimReductionIm,SpatialBW,RangeBW,option) 
% option =0, three channels edison seg
% option !=0 , one channel edison seg
[Imrow,Imcol,Imdepth] = size(TestIm);
if isequal(option,0)
[labels,modes,regSize] = edison_wrapper(DimReductionIm , @RGB2Luv,...
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


else
S = msseg(DimReductionIm,SpatialBW,RangeBW)  ;                         
v= S(:);
n= (v-min(v))/(max(v)-min(v));
S_norm= (reshape(n,[Imrow Imcol]));

%%imresize
SegIm =(uint16(im2uint8(S_norm)));
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
end

end