close all 
clear all
clc

load('classifier')

camera=webcam;
while true
    img=camera.snapshot;
    imshow(img);
    [labelIdx, scores] = predict(categoryClassifier, img);
    
    k = 0;
    for i = 1:length(scores)
        if scores(i)> -0.15        
            k = 1;
        end
    end
    if k == 1
        categoryClassifier.Labels(labelIdx)
        scores
    else
        'nothing'
        scores
    end
end
