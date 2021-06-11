
% col1= "Validation Accuracy";
% col2= "Test1 Accuracy";
% col3= "Test2 Accuracy";
% col4 = "Macro Precision";
% col5 = "Weighted Precision";
% col6 = "Macro Recall";
% col7 = "Weighted Recall";
% col8 = "Macro Fscore";
% col9 = "Weighted Fscore";
% col10 = "Minimum Precision class";
% col11 = "Minimum Recall class";
% col12 = "Minimum Fscore class";
% for i = 0 : 21
%     tab(1,12*i+1)=col1;
%     tab(1,12*i+2)=col2;
%     tab(1,12*i+3)=col3;
%     tab(1,12*i+4)=col4;
%     tab(1,12*i+5)=col5;
%     tab(1,12*i+6)=col6;
%     tab(1,12*i+7)=col7;
%     tab(1,12*i+8)=col8;
%     tab(1,12*i+9)=col9;
%     tab(1,12*i+10)=col10;
%     tab(1,12*i+11)=col11;
%     tab(1,12*i+12)=col12;
% end
% T = table(tab);

%load TA2.mat;


% classnames = {'0';'1'; '10';'11';'12';'13';'14';'15';'16';'17';'18';'19';...
%     '2';'20';'21';'22';'23';'24';'25';'26';'27';'28';'29';...
%     '3';'30';'31';'32';'33';'34';'35';'36';...
%     '4';'5';'6';...
%     '7';'8';'9';};
%  classnames = {'aa';'bho';'bishorgo';'ga';'la';'po';'rri';'ta';'th';'tho'};
% % classnames = {'ae';'ah';'b';'bsro';'cha';'chha';'chndro';'da';'dda';'ddha';'dha';'eio';...
% %                'fa';'ga';'gha';'ha';'i';'ja';'jha';'k';'kha';'la';'ma';'na';'o';...
% %                'oha';'oi';'onsor';'ou';'pa';'rri';'sa';'ta';'tha';'tta';'ttha';'u';'umo'};
n=1;
classnames = {'COVID-19';'NORMAL';'Pneumonia';};

%%
net = assembleNetwork(lgraph_3);

%%
inputSize = net.Layers(1).InputSize;
imdsTrain.ReadFcn = @(loc)imresize(imread(loc),inputSize(1:2));
%imdsTest.ReadFcn = @(loc)imresize(imread(loc),inputSize(1:2));
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'ColorPreprocessing','gray2rgb');
%augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest,'ColorPreprocessing','gray2rgb');
%%
layer = 'avg1';
clear featuresTest featuresTrain;
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
disp("feature Train Done");
%%
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');
disp("feature Test done");
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
%%
p1=1;%p=1 for cost unchanged,p=2 for cost changed

% clear features;
% features(1:4727,1:size(featuresTrain,2)) = featuresTrain(1:4727,1:size(featuresTrain,2));
% features(4728:7879,1:size(featuresTrain,2)) = featuresTest(1:3152,1:size(featuresTrain,2));
% [Y,loss] = tsne(features);
% featuresTrain = Y(1:4727,1:2);
% featuresTest = Y(4728:7879,1:2);
classnames = {'COVID-19';'NORMAL';'Pneumonia';};
clear colName;
colName = getColNames(size(featuresTrain,2));
    for j = 9 : 9 
        if j == 1
        [classifier, validationAccuracy] = costunchanged1(featuresTrain, YTrain,p1,classnames,colName);
            figname = '1_FineTree';
        elseif j == 2
            [classifier, validationAccuracy] = costunchanged2(featuresTrain, YTrain,p1,classnames,colName);
            figname = '2_MediumTree';
            elseif j == 3
            [classifier, validationAccuracy] = costunchanged3(featuresTrain, YTrain,p1,classnames,colName);
            figname = '3_CoarseTree';
            elseif j == 4
            [classifier, validationAccuracy] = costunchanged4(featuresTrain, YTrain,p1,classnames,colName);
            figname = '4_LinearDiscriminant';
            elseif j == 5
            %[classifier, validationAccuracy] = costunchanged5(featuresTrain, YTrain,p1,classnames,colName);
            %figname = '5_NaiveBayes';
            continue;
            elseif j == 6
            [classifier, validationAccuracy] = costunchanged6(featuresTrain, YTrain,p1,classnames,colName);
            figname = '6_LinearSVM';
            elseif j == 7
                %continue;
            [classifier, validationAccuracy] = costunchanged7(featuresTrain, YTrain,p1,classnames,colName);
            
            figname = '7_QuadraticSVM';
            elseif j == 8
            %[classifier, validationAccuracy] = costunchanged8(featuresTrain, YTrain,p1,classnames,colName);
            %figname = '8_CubicSVM';
            continue;
        elseif j == 9
            [classifier, validationAccuracy] = costunchanged9(featuresTrain, YTrain,p1,classnames,colName);
            figname = '9_MedGaussSVM';
        elseif j == 10
            [classifier, validationAccuracy] = costunchanged10(featuresTrain, YTrain,p1,classnames,colName);
            figname = '10_CoarseGAussSVM';
        elseif j == 11
            [classifier, validationAccuracy] = costunchanged11(featuresTrain, YTrain,p1,classnames,colName);
            figname = '11_FineKNN';
            elseif j == 12
            [classifier, validationAccuracy] = costunchanged12(featuresTrain, YTrain,p1,classnames,colName);
            figname = '12_MediumKNN';
            elseif j == 13
            [classifier, validationAccuracy] = costunchanged13(featuresTrain, YTrain,p1,classnames,colName);
            figname = '13_CoarseKNN';
            elseif j == 14
            [classifier, validationAccuracy] = costunchanged14(featuresTrain, YTrain,p1,classnames,colName);
            figname = '14_CosineKNN';
            elseif j == 15
                continue;
%             [classifier, validationAccuracy] = costunchanged15(featuresTrain, YTrain,p1,classnames,colName);
%             figname = '15_CubicKNN';
            elseif j == 16
            [classifier, validationAccuracy] = costunchanged16(featuresTrain, YTrain,p1,classnames,colName);
            figname = '16_WeightedKNN';
            elseif j == 17
            %[classifier, validationAccuracy] = costunchanged17(featuresTrain, YTrain,p1,classnames,colName);
            %figname = '17_BoostedTree';
            continue;
            elseif j == 18
            [classifier, validationAccuracy] = costunchanged18(featuresTrain, YTrain,p1,classnames,colName);
            figname = '18_BaggedTree';
            elseif j == 19
            [classifier, validationAccuracy] = costunchanged19(featuresTrain, YTrain,p1,classnames,colName);
            figname = '19_SubspaceDiscriminant';
            elseif j == 20
                continue;
            %[classifier, validationAccuracy] = costunchanged20(featuresTrain, YTrain,p1,classnames,colName);
            %figname = '20_SubspaceKNN';
            elseif j == 21
                continue;
            %[classifier, validationAccuracy] = costunchanged21(featuresTrain, YTrain,p1,classnames,colName);
            %figname = '21_RusBoostedTree';
        end
        [YPred,score] = classifier.predictFcn(featuresTest);
        TestAccuracy1 = mean(YPred == YTest);

        %[YPred2,score] = classifier.predictFcn(featuresTest2);
        %TestAccuracy2 = mean(YPred2 == YTest2);
        
        YPred2 = YPred;
        TestAccuracy2=0;
        
        Target_row = transpose(grp2idx(YTest));
        Target_encoded=full(ind2vec(Target_row));
        score_transpose = score';
        [tpr,fpr,thresholds] = roc(Target_encoded,score_transpose);

        [c,cm,ind,per] = confusion(Target_encoded,score_transpose);
        numClass = countcats(YTest);
        macro_precision_sum =0;
        macro_sense_sum =0;
        macro_spec_sum =0;
        macro_recall_sum = 0;
        macro_fscore_sum =0;
        weighted_precision_sum = 0;
        weighted_recall_sum = 0;
        weighted_fscore_sum =0;
        weighted_sense_sum =0;
        weighted_spec_sum =0;
        for i = 1 : size(per,1)
            p(i) = per(i,3)/(per(i,3)+per(i,2));
            r(i) = per(i,3)/(per(i,3)+per(i,1));
            f(i) = ( 2*p(i)*r(i))/(p(i)+r(i));
            sn(i) = per(i,3)/(per(i,3)+per(i,1));
            sp(i) = per(i,4)/(per(i,4)+per(i,2));
            macro_precision_sum = macro_precision_sum + p(i);
            macro_recall_sum = macro_recall_sum + r(i);
            macro_fscore_sum = macro_fscore_sum + f(i);
            macro_sense_sum = macro_sense_sum + sn(i);
            macro_spec_sum = macro_spec_sum + sp(i);
            weighted_precision_sum = weighted_precision_sum + numClass(i) * p(i);
            weighted_recall_sum = weighted_recall_sum + numClass(i) * r(i);
            weighted_fscore_sum = weighted_fscore_sum + numClass(i) * f(i);
            weighted_sense_sum = weighted_sense_sum + numClass(i) * sn(i);
            weighted_spec_sum = weighted_spec_sum + numClass(i) * sp(i);
        end
            macro_precision = macro_precision_sum / size(per,1);
            weighted_precision = weighted_precision_sum / sum(countcats(YTest));
            macro_recall = macro_recall_sum /size(per,1);
            weighted_recall = weighted_recall_sum / sum(countcats(YTest));
            macro_fscore = macro_fscore_sum / size(per,1);
            weighted_fscore = weighted_fscore_sum / sum(countcats(YTest));
            weighted_sense = weighted_sense_sum / sum(countcats(YTest));
            weighted_spec = weighted_spec_sum / sum(countcats(YTest));
            [minPrec,PrecClass] = min(p);
            [minRec,RecClass] = min(r);
            [minF,FClass] = min(f);
            if p1==1
            TA2(n,12*(j-1)+1) = validationAccuracy;
            TA2(n,12*(j-1)+2) = TestAccuracy1;
            TA2(n,12*(j-1)+3) = TestAccuracy2;
            TA2(n,12*(j-1)+4) = macro_precision;
            TA2(n,12*(j-1)+5) = weighted_precision;
            TA2(n,12*(j-1)+6) = macro_recall;
            TA2(n,12*(j-1)+7) = weighted_recall;
            TA2(n,12*(j-1)+8) = macro_fscore;
            TA2(n,12*(j-1)+9) = weighted_fscore;
            TA2(n,12*(j-1)+10) = weighted_sense;
            TA2(n,12*(j-1)+11) = weighted_spec;
            TA2(n,12*(j-1)+12) = FClass;
            elseif p1==2
                TA3(n,12*(j-1)+1) = validationAccuracy;
            TA3(n,12*(j-1)+2) = TestAccuracy1;
            TA3(n,12*(j-1)+3) = TestAccuracy2;
            TA3(n,12*(j-1)+4) = macro_precision;
            TA3(n,12*(j-1)+5) = weighted_precision;
            TA3(n,12*(j-1)+6) = macro_recall;
            TA3(n,12*(j-1)+7) = weighted_recall;
            TA3(n,12*(j-1)+8) = macro_fscore;
            TA3(n,12*(j-1)+9) = weighted_fscore;
            TA3(n,12*(j-1)+10) = weighted_sense;
            TA3(n,12*(j-1)+11) = weighted_spec;
            TA3(n,12*(j-1)+12) = FClass;
            end
            %misClass(size(misClass,1)+1,:) = (YPred' ~= YTest');
            %plotconfusion(YTest,YPred);
            cm = confusionchart(YTest,YPred);
            cm.ColumnSummary = 'column-normalized';
            cm.RowSummary = 'row-normalized';
            %sortClasses(cm,'descending-diagonal');
            %cm.Normalization = 'absolute';
            %hold on;
            %saveas(gcf,figname,'pdf');
            %hold off;
            %plotroc(Target_encoded,score_transpose);
            %hold on;
            %saveas(gcf,figname,'jpg');
            %hold off;
            disp(j);
            
%             clear classifier validationAccuracy TestAccuracy1 TestAccuracy2 macro_precision_sum p r f ...
%                 macro_precision weighted_precision macro_recall weighted_recall weighted_recall ...
%                 macro_fscore weighted_fscore PrecClass RecClass FClass YPred YPred2 score
    end
    %save TA2;