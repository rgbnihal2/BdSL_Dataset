clc;
clear;

%%
imagepath = fullfile('Train');
imds = imageDatastore(imagepath,'IncludeSubfolders',true,'LabelSource','Foldernames');

%%
imds.ReadFcn = @(loc)imresize(imread(loc),[224,224]);

%%
[trainDS,valDS] = splitEachLabel(imds,0.7,0.3,'randomized');

%%
% opts = trainingOptions('sgdm', ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropFactor',0.2, ...
%     'LearnRateDropPeriod',5, ...
%     'MaxEpochs',20, ...
%     'MiniBatchSize',64, ...
%     'Plots','training-progress')

opts = trainingOptions('sgdm','InitialLearnRate',0.001, ...
    'ValidationData',valDS,...
    'ValidationFrequency',5, ...
    'ValidationPatience',5, ...
    'MaxEpochs',300, ...
    'MiniBatchSize',64, ...
    'Verbose',true, ...
    'Plots','training-progress');

%%
network = trainNetwork(trainDS,google13,opts);

%%
testPath = fullfile('train2');
imdsTest = imageDatastore(testPath,'IncludeSubfolders',true,'LabelSource','Foldernames');
imdsTest.ReadFcn = @(loc)imresize(imread(loc),[224,224]);
%%
YPred = classify(network,imdsTest);
YValidation = imdsTest.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)