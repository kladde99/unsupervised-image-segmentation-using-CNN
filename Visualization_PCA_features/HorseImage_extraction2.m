% Horse Images_extraction, read the raw image data and convert it to matlab
% format
clc;
clear all;
close all;

%  transfer jpg image to matlab image data

files = dir(fullfile('/Users/Ben/Documents/MATLAB/horse_feature_analysis/figure_ground/','*.jpg'));
lengthfiles = length(files);

% imdb.images.data = [];
horse_image = cell(0);
for i=1: lengthfiles

image = imread(['/Users/Ben/Documents/MATLAB/horse_feature_analysis/figure_ground/',files(i).name]);
horseImage = imresize(image,[144 160]);
horse_image{i} = imbinarize(horseImage);
end
groundTruth = horse_image;
save('groundTruth_binary.mat','groundTruth');