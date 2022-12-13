close all
clear all
clc

%This script helps to train a SURF model using a toolbox from MatLab

tic
%unzip('imageSet.zip');
%In order to use your folder you must change imageSet with the path in your
%PC, different lasses must be separated by folders in imageSet folder.
imds = imageDatastore('imageSet','IncludeSubfolders',true,'LabelSource','foldernames');
tbl = countEachLabel(imds);

[trainingSet, validationSet] = splitEachLabel(imds, 0.6, 'randomize');
bag = bagOfFeatures(trainingSet);
toc

categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);
confMatrix = evaluate(categoryClassifier, trainingSet);

confMatrix = evaluate(categoryClassifier, validationSet);
mean(diag(confMatrix))

%It creates a .mat file in the current folder with 'Classifier.mat' name
save('Classifier.mat','categoryClassifier','imds','trainingSet','validationSet','bag')
