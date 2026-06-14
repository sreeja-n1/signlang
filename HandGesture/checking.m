clc;
clear all;
close all;
gestureNames = {'fist','thumbs-up', 'nope', 'no-sign', 'hiee'};
numGestures = length(gestureNames);
imd = imageDatastore(gestureNames, 'LabelSource', 'foldernames',            ...
    'IncludeSubfolders', true);
imd.ReadFcn = @(filename)imresize(imread(filename), [227 227]);
[imdTrain, imdValidation] = splitEachLabel(imd, 0.8, 'randomized');
disp(imdTrain.Labels);
disp('Training dataset:');
disp(countEachLabel(imdTrain));
disp('Validation dataset:');
disp(countEachLabel(imdValidation));
