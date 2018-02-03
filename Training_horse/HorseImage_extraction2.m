% HorseImage_extraction

close all;
clear all;
clc;

%  transfer jpg image to matlab image data

files = dir(fullfile('/Users/Ben/Documents/MATLAB/horse/weizmann_horse_dataset/','*.jpg'));
lengthfiles = length(files);

imdb.images.data = [];

for i=1: lengthfiles
    
    horse_image{i} = imread(['/Users/Ben/Documents/MATLAB/horse/weizmann_horse_dataset/',files(i).name]);
     horse_image{i} = imresize(horse_image{i},[144 160]);
     horse_image{i} = im2single(horse_image{i});
     imdb.images.data = cat(4,imdb.images.data,horse_image{i});
    
end

imdb.images.labels = imdb.images.data ;
imdb.images.set=ones(1,326);
imdb.images.set(:,randperm(326,66)) = 2*ones(1,66);  % randomly choose validation set

save('imdb.mat');