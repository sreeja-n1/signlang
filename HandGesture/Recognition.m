clc;
clear all;
close all;
load('trainedCustomCNN.mat'); 
gestureLabels = {'fist','thumbs-up', 'nope', 'no-sign', 'hiee'};
c = webcam;
x = 100; 
y = 100;
width = 200;
height = 200;
bboxes = [x, y, width, height];
hFig = figure('Name','Real-Time Gesture Recognition','NumberTitle','off');
axis off;  
while true
    e = c.snapshot;
    imshow(e);
    hold on;
    rectangle('Position', bboxes, 'EdgeColor', 'r', 'LineWidth', 2);  
    hold off;
    croppedImage = imcrop(e, bboxes);
    croppedImage = imresize(croppedImage, [227 227]); 
    [label, score] = classify(trainedNet, croppedImage);
    title(['Predicted Gesture: ' char(label)]);
    disp(['Prediction: ', char(label), ' with confidence: ',                 ...
        num2str(max(score))]);
    drawnow;
    if ~ishandle(hFig)
        break;
    end
end
clear c;


