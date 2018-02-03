% see the struct of cae

close all;
clear all;
clc;
% load toolbox and dataset
run('/Users/Ben/Documents/MATLAB/matconvnet-new-solvers/matlab/vl_setupnn');
load('/Users/Ben/Documents/MATLAB/horse2_works/imdb.mat');

net = horse_cae_init();

res = vl_simplenn(net, imdb.images.data(:,:,:,1)) ;
