%% devide the dataset into  training data validation data and test data

close all;
clear all;
clc;
imdb.images.data = [];
%  transfer jpg image to matlab image data

files = dir(fullfile('/Users/Ben/Documents/MATLAB/BSDS500/data/images/train/','*.jpg'));

lengthfiles = length(files);

for i=1: lengthfiles
    
    bsds300_image{i} = imread(['/Users/Ben/Documents/MATLAB/BSDS500/data/images/train/',files(i).name]);
     [imrow,imcol,depth] = size(bsds300_image{1, i}); 
     if  imrow < imcol 
         bsds300_image{1, i} = permute(bsds300_image{1, i},[2 1 3]);
     end
     bsds300_image{i} = imresize( bsds300_image{i},[192 128]);
     bsds300_image{i} = im2single(bsds300_image{i});
     imdb.images.data = cat(4,imdb.images.data,bsds300_image{i});
    
end


files = dir(fullfile('/Users/Ben/Documents/MATLAB/BSDS500/data/images/val/','*.jpg'));
lengthfiles = length(files);

for i=1: lengthfiles
    
    bsds300_image{i} = imread(['/Users/Ben/Documents/MATLAB/BSDS500/data/images/val/',files(i).name]);
     [imrow,imcol,depth] = size(bsds300_image{1, i}); 
     if  imrow < imcol 
         bsds300_image{1, i} = permute(bsds300_image{1, i},[2 1 3]);
     end
     bsds300_image{i} = imresize( bsds300_image{i},[192 128]);
     bsds300_image{i} = im2single(bsds300_image{i});
     imdb.images.data = cat(4,imdb.images.data,bsds300_image{i});
    
end

imdb.images.labels = imdb.images.data ;
imdb.images.set=ones(1,300);
imdb.images.set(:,randperm(300,60)) = 2*ones(1,60);  % randomly choose validation set

save('imdb.mat');
