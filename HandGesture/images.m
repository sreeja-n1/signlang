clc;
clear all;
close all;
warning off;
c = webcam; 
gestureNames = {'fist','thumbs-up', 'nope', 'no-sign', 'hiee'};
numImagesPerGesture = 50;  
x = 100;
y = 100;
width = 200;
height = 200;
bboxes = [x, y, width, height];
for g = 1:length(gestureNames)
    if ~exist(gestureNames{g}, 'dir')
        mkdir(gestureNames{g});  
    end
end
for g = 1:length(gestureNames)
    disp(['Capturing images for ' gestureNames{g}]);

    for temp = 1:numImagesPerGesture
        e = c.snapshot;
        figure;
        imshow(e);  
        hold on;
        rectangle('Position', bboxes, 'EdgeColor', 'r', 'LineWidth', 2); 
        hold off;
        croppedImage = imcrop(e, bboxes);  
        croppedImage = imresize(croppedImage, [227 227]); 
        filename = fullfile(gestureNames{g}, strcat(gestureNames{g}, '_', ...
            num2str(temp, '%04d'), '.bmp'));
        imwrite(croppedImage, filename);  
        disp(['Captured image ' num2str(temp) ' for ' gestureNames{g}]);
        drawnow;
    end
end
clear c;          