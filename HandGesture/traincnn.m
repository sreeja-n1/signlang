clc;
clear all;
close all;
gestureNames = {'fist','thumbs-up', 'nope', 'no-sign', 'hiee'};
imds = imageDatastore(gestureNames, 'LabelSource', 'foldernames',           ...
    'IncludeSubfolders', true);
imds.ReadFcn = @(filename)imresize(imread(filename), [227 227]);
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');
layers=[imageInputLayer([227 227 3],'Name','input','Normalization','none')  
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'BN1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1') 
    
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'BN2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')  
   
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'BN3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3') 
  
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'BN4')
    reluLayer('Name', 'relu4')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool4')
   

    convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'conv5')
    batchNormalizationLayer('Name', 'BN5')
    reluLayer('Name', 'relu5')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool5') 
    
    
    fullyConnectedLayer(1024, 'Name', 'fc1')  
    reluLayer('Name', 'relu_fc1')
    
   
    fullyConnectedLayer(512, 'Name', 'fc2')  
    reluLayer('Name', 'relu_fc2')
    fullyConnectedLayer(5, 'Name', 'fc3')  
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 20, ...  
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'Verbose', false, ...
    'Plots', 'training-progress');
disp('Training the custom CNN...');
trainedNet = trainNetwork(imdsTrain, layers, options);
save('trainedCustomCNN.mat', 'trainedNet');
disp('Model training completed.');
