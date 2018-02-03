%% HorseImage_extraction and this version is better

close all;
clear all;
clc;

imdb.images.data = [];
%  transfer jpg image to matlab image data

files = dir(fullfile('/Users/Ben/Documents/MATLAB/bsds500works/data/images/train/','*.jpg'));

lengthfiles = length(files);

for i=1: lengthfiles
    
    bsds500_image{i} = imread(['/Users/Ben/Documents/MATLAB/bsds500works/data/images/train/',files(i).name]);
     [imrow,imcol,depth] = size(bsds500_image{1, i}); 
     if  imrow < imcol 
         bsds500_image{1, i} = permute(bsds500_image{1, i},[2 1 3]);
     end
     bsds500_image{i} = imresize( bsds500_image{i},[192 128]);
     bsds500_image{i} = im2single(bsds500_image{i});
     bsds500_image{i} = vl_imsmooth(bsds500_image{i},5);
     imdb.images.data = cat(4,imdb.images.data,bsds500_image{i});
    
end

files = dir(fullfile('/Users/Ben/Documents/MATLAB/bsds500works/data/images/test/','*.jpg'));
lengthfiles = length(files);
for i=1: lengthfiles
    
    bsds500_image{i} = imread(['/Users/Ben/Documents/MATLAB/bsds500works/data/images/test/',files(i).name]);
     [imrow,imcol,depth] = size(bsds500_image{1, i}); 
     if  imrow < imcol 
         bsds500_image{1, i} = permute(bsds500_image{1, i},[2 1 3]);
     end
      bsds500_image{i} = imresize( bsds500_image{i},[192 128]);
     bsds500_image{i} = im2single(bsds500_image{i});
     bsds500_image{i} = vl_imsmooth(bsds500_image{i},5);
     imdb.images.data = cat(4,imdb.images.data,bsds500_image{i});
    
end

files = dir(fullfile('/Users/Ben/Documents/MATLAB/bsds500works/data/images/val/','*.jpg'));
lengthfiles = length(files);

for i=1: lengthfiles
    
    bsds500_image{i} = imread(['/Users/Ben/Documents/MATLAB/bsds500works/data/images/val/',files(i).name]);
     [imrow,imcol,depth] = size(bsds500_image{1, i}); 
     if  imrow < imcol 
         bsds500_image{1, i} = permute(bsds500_image{1, i},[2 1 3]);
     end
     bsds500_image{i} = imresize( bsds500_image{i},[192 128]);
     bsds500_image{i} = im2single(bsds500_image{i});
     bsds500_image{i} = vl_imsmooth(bsds500_image{i},5);
     imdb.images.data = cat(4,imdb.images.data,bsds500_image{i});
    
end


imdb.images.labels = imdb.images.data ;
imdb.images.set=ones(1,500);
imdb.images.set(:,randperm(500,100)) = 2*ones(1,100);  % randomly choose validation set
% imageMean = mean(imdb.images.data(:)) ;
% imdb.images.data = imdb.images.data - imageMean ;

save('imdb_noise5.mat','imdb');