clc;
clear;

%%
imagepath = fullfile('E:\MATLAB\Sign Language Recognition\BdSL datasets\dataset Rafi\RESIZED_DATASET');
imds = imageDatastore(imagepath,'IncludeSubfolders',true,'LabelSource','Foldernames');
imds = shuffle(imds);

%%
[trainDS,valDS] = splitEachLabel(imds,0.9,0.1,'randomized');

%%
net = lgraph_2;% from deep network designer
inputSize = net.Layers(1).InputSize;
trainDS.ReadFcn = @(loc)imresize(imread(loc),inputSize(1:2));
valDS.ReadFcn = @(loc)imresize(imread(loc),inputSize(1:2));
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true);
    %'RandXTranslation',pixelRange, ...
    %'RandYTranslation',pixelRange, ...
    %'RandXScale',scaleRange, ...
    %'RandYScale',scaleRange
    
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainDS, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),valDS);

%

% inputSize = net.Layers(1).InputSize;
% 
% if isa(net,'SeriesNetwork') 
%   lgraph = layerGraph(net.Layers); 
% else
%   lgraph = layerGraph(net);
% end 
%%
layers = net.Layers;
connections = net.Connections;

layers(1:18) = freezeWeights(layers(1:18));
lgraph = createLgraphUsingConnections(layers,connections);

%%
checkpointPath = pwd;
% 
% opts = trainingOptions('sgdm', ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropFactor',0.2, ...
%     'LearnRateDropPeriod',5, ...
%     'MaxEpochs',20, ...
%     'MiniBatchSize',64, ...
%     'Plots','training-progress')

opts = trainingOptions('adam','InitialLearnRate',0.001, ...
    'ValidationData',augimdsValidation,...
    'ValidationFrequency',31, ...
    'MaxEpochs',1, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'MiniBatchSize',32, ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'CheckpointPath',checkpointPath);

%
network = trainNetwork(augimdsTrain,lgraph,opts);

%%
testPath = fullfile('E:\MATLAB\Sign Language Recognition\BdSL datasets\dataset Rafi\RESIZED_TESTING_DATA');
imdsTest = imageDatastore(testPath,'IncludeSubfolders',true,'LabelSource','Foldernames');
imdsTest.ReadFcn = @(loc)imresize(imread(loc),inputSize(1:2));

YPred = classify(network,imdsTest);
YValidation = imdsTest.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)