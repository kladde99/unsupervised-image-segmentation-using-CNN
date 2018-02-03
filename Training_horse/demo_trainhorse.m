% a demo script for training network and visualize the image reconstrction
% and CAE layers from Weizmann horse dataset.
 
%% train CAE network under Weizmann horse dataset
% first change the folder to cd ../quellcode/;
horse_CAE_train

%% visualize image reconstrcuction
horse_eva_recon

%% visualize the feature maps from the code layer 
horse_eva_code_layer

%%  show the featuresmaps from the conv layers of the pretrained imagenet model
horse_premodel