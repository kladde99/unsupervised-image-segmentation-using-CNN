% a demo script for training network and visualize the image reconstrction
% and CAE layers from BSDS500.
 
%% train CAE network under BSDS500 dataset
% first change the folder to cd ../quellcode/;
BSDS500_CAE_train

%% visualize image reconstrcuction
bsds_eva_recon

%% visualize the feature maps from the code layer 
bsds_eva_codelayer

%%  show the featuresmaps from the all theconv layers of CAE1/CAE2
show_all_featuremaps

%% visualize the feature maps of the conv layers from imagenet pretrained model
PretrainModelFeatureAnalysis

%% feature transform and visualization
FeatureAnalysis

%% visualization of PCA_transformed features using CAE1/2 under Weizmann horse dataset
main_feature_analysis