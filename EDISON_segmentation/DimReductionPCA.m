function [DimReductionIm]= DimReductionPCA(code_layer_transfer,TestIm,num,flag)

% input: code: code layer features
%        num: number of the Dim you want to reduce to.
%        flag =0 without RGB; flag !=0 with RGB

if isequal(flag,0)
NewChannels = code_layer_transfer;

%  normalization
NewChannels_vector = NewChannels(:);
NewChannels_norm = (NewChannels_vector-min(NewChannels_vector))/...
(max(NewChannels_vector)-min(NewChannels_vector));

FeatureVector = reshape(NewChannels_norm,size(NewChannels,1)*size(NewChannels,2),...
size(NewChannels,3));
[coeff,score,latent] = pca(FeatureVector);
TransferIm = FeatureVector*coeff;

TransferIm_new = im2uint8(reshape(TransferIm,size(NewChannels,1),size(NewChannels,2),...
size(NewChannels,3)));

if   isequal(num,1)   
   DimReductionIm  = TransferIm_new (:,:,1);
else
    if isequal(num,3)
       DimReductionIm  = TransferIm_new (:,:,1:3); 

    else 
        if isequal(num,2)
            DimReductionIm  = TransferIm_new (:,:,1:2); 
    else
        error('number must be 1 or 2 or 3');
    end
    end
end

else
NewChannels = cat(3,TestIm, code_layer_transfer);

%  normalization
NewChannels_vector = NewChannels(:);
NewChannels_norm = (NewChannels_vector-min(NewChannels_vector))/...
             (max(NewChannels_vector)-min(NewChannels_vector));

FeatureVector = reshape(NewChannels_norm,size(NewChannels,1)*size(NewChannels,2),...
            size(NewChannels,3));
[coeff,score,latent] = pca(FeatureVector);
TransferIm = FeatureVector*coeff;

TransferIm_new = im2uint8(reshape(TransferIm,size(NewChannels,1),size(NewChannels,2),...
         size(NewChannels,3)));

if   isequal(num,1)   
       DimReductionIm  = TransferIm_new (:,:,1);
else
    if isequal(num,3)
       DimReductionIm  = TransferIm_new (:,:,1:3); 
    else
        if isequal(num,2)
          DimReductionIm  = TransferIm_new (:,:,1:2);  
    else
        error('number must be 1 or 2 or 3');
        end
    end
end
end