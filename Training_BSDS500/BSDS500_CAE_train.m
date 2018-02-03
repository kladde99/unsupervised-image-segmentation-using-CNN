%% training CAE on BSDS 500

close all;
clear all;
clc;
%% load images, toolbox
run('../quellcode/matconvnet-new-solvers/matlab/vl_setupnn');
load('../quellcode/Training_BSDS500/imdb.mat');
%% different strcut of cnn
 net = bsds_cae_init1(); % load the cae 1
 %  net = bsds_cae_init2(); % uncomment to switch to train CAE 2
%% add L2 loss
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
% trainOpt.continue = false;
net = cnn_train(net, imdb, @getBatch, trainOpts) ;
%% Deploy: remove loss
net.layers(end) = [] ;