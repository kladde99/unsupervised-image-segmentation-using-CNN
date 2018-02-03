%% training CAE from Weizmann horse dataset 

close all;
clear all;
clc;
%% initialize the matconv toolbox and image dataset
run('../quellcode/matconvnet-new-solvers/matlab/vl_setupnn');
rng(0);
load('../quellcode/Training_horse/imdb.mat');
%% load struct of cae
 net = horse_cae_init();

%% add loss function
net = addCustomLossLayer(net, @l2LossForward, @l2LossBackward) ;

%% Train
trainOpts.expDir = 'data/test' ;
trainOpts.gpus = [] ;
% Uncomment for GPU training:
%trainOpts.expDir = 'data/text-small-gpu' ;
%trainOpts.gpus = [1] ;
trainOpts.batchSize = 10 ;
trainOpts.learningRate = 0.01 ;
trainOpts.plotDiagnostics = false ;
%trainOpts.plotDiagnostics = true ; % Uncomment to plot diagnostics
trainOpts.numEpochs = 1000;
trainOpts.errorFunction = 'none' ;

net = cnn_train(net, imdb, @getBatch, trainOpts) ;

%% Deploy: remove loss
net.layers(end) = [] ;