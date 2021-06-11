clc;
clear;

%%
imagepath = fullfile('Training');
imdsTrain = imageDatastore(imagepath,'IncludeSubfolders',true,'LabelSource','Foldernames');
%imdsTrain = shuffle(imdsTrain);
%[imdsTrain,imdsTest] = splitEachLabel(imds,0.6,'randomized');
imagepath = fullfile('Test');
imdsTest = imageDatastore(imagepath,'IncludeSubfolders',true,'LabelSource','Foldernames');

%%
net = lgraph_1 ;

%%
inputSize = net.Layers(1).InputSize;
% % Data Augumentation
    augmenter = imageDataAugmenter( ...
        'RandRotation',[-5 5],'RandXReflection',1,...
        'RandYReflection',1,'RandXTranslation',[-10 10],'RandYTranslation',[-5 5],'RandScale',[0.5 2]);
%     
%     % Resizing all training images to [224 224] for ResNet architecture
%     auimds = augmentedImageDatastore([224 224],imdsTrain,'DataAugmentation',augmenter);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'DataAugmentation',augmenter);
%augimdsTestaugimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTest,'DataAugmentation',augmenter);
%%
layer = 'avg_pool';
imdsTrain.ReadFcn = @(loc)imresize(imread(loc),inputSize(1:2));
imdsTest.ReadFcn = @(loc)imresize(imread(loc),inputSize(1:2));
%clear featuresTest featuresTrain;
featuresTrain = activations(net,imdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,imdsTest,layer,'OutputAs','rows');
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
%%
% %%
% classifier = trainedModel;%this model is exported from classification learner app
% YPred = classifier.predictFcn(featuresTest);
% accuracy = mean(YPred == YTest)
% 
% %%
% imagepath2 = fullfile('train2');
% imdsTest2 = imageDatastore(imagepath2,'IncludeSubfolders',true,'LabelSource','Foldernames');
% augimdsTest2 = augmentedImageDatastore(inputSize(1:2),imdsTest2);
% featuresTest2 = activations(net,augimdsTest2,layer,'OutputAs','rows');
% YTest2 = imdsTest2.Labels;
% %%
% [YPred2,score] = classifier.predictFcn(featuresTest2);
% accuracy = mean(YPred2 == YTest2)
% plotconfusion(YTest2,YPred2)
% 
% %%
% % Evaluation Metrics
% Target_row = transpose(grp2idx(YTest2));
% Target_encoded=full(ind2vec(Target_row));
% score_transpose = score';
% [tpr,fpr,thresholds] = roc(Target_encoded,score_transpose);
% figure;
% plotroc(Target_encoded,score_transpose);
% %%
% %precision and recall
% [c,cm,ind,per] = confusion(Target_encoded,score_transpose);
% numClass = countcats(YTest2);
% macro_precision_sum =0;
% macro_recall_sum = 0;
% macro_fscore_sum =0;
% weighted_precision_sum = 0;
% weighted_recall_sum = 0;
% weighted_fscore_sum =0;
% for i = 1 : length(per)
%     p(i) = per(i,3)/(per(i,3)+per(i,2));
%     r(i) = per(i,3)/(per(i,3)+per(i,1));
%     f(i) = ( 2*p(i)*r(i))/(p(i)+r(i));
%     macro_precision_sum = macro_precision_sum + p(i);
%     macro_recall_sum = macro_recall_sum + r(i);
%     macro_fscore_sum = macro_fscore_sum + f(i);
%     weighted_precision_sum = weighted_precision_sum + numClass(i) * p(i);
%     weighted_recall_sum = weighted_recall_sum + numClass(i) * r(i);
%     weighted_fscore_sum = weighted_fscore_sum + numClass(i) * f(i);
%     
% end
%     macro_precision = macro_precision_sum / length(per)
%     weighted_precision = weighted_precision_sum / sum(countcats(YTest2))
%     macro_recall = macro_recall_sum /length(per)
%     weighted_recall = weighted_recall_sum / sum(countcats(YTest2))
%     macro_fscore = macro_fscore_sum / length(per)
%     weighted_fscore = weighted_fscore_sum / sum(countcats(YTest2))
% %%
% %AUC,ROC
% A=grp2idx(YTest2);
% B=grp2idx(YPred2);
% for i = 1 : length(per)
%     [X(:,i),Y(:,i),T(:,i),AUC(:,i)] = perfcurve(A,B,i);
% end
% min(AUC)