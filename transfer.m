alexX = alexnet;
layers = alexX.Layers;

layers(23) = fullyConnectedLayer(5);
layers(25) = classificationLayer;

%In here you should replace myImages3 file with the path of your
%train data that includes face images. 
%Let's say that the folder name is Faces
%In the folder Faces you should have subfolders which include the individuals's faces
%For example; in face folder you have a folder named Alice and that folder
%includes Alice's face photos. And another folder named John...
%After that you run the code, you change the last layer of the network. 
%1000 photos for each individual is recomended but in my opinion for little
%tests 300 per each is enough.
%You can create the images easily with your phones multiple shooting mode.
allImages = imageDatastore('myImages3', 'IncludeSubFolders', true, 'LabelSource', 'foldernames');
[trainingImages, testImages] = splitEachLabel(allImages, 0.8, 'randomize');

opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 20, 'MiniBatchSize', 64);

%In here myNet3 will be your trained network. You can name it the way you
%like. When its created, don't forget to save it. If you don't you will
%have to do the same thing and wait for the long process again.
myNet3 = trainNetwork(trainingImages, layers, opts);

predictedLabels = classify(myNet3, testImages);
accuracy = mean(predictedLabels == testImages.Labels);

%After the process, alexnet will make the predictions as the names of the
%subfolders.
