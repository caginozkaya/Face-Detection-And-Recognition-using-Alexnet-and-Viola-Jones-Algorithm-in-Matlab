
%ATTENTION
%Before runnig the code, you have to load the matrice that created with the
%training step. If you hadn't done that go ahead and do it first.
%Open transfer.m file and follw the instructions.

while true
prompt = 'Enter the image:';
pic = input(prompt, 's');
A = imread(pic);

FaceDetector = vision.CascadeObjectDetector();

BBOX = step(FaceDetector, A);






B = insertObjectAnnotation(A, 'rectangle', BBOX, 'face');
%B = insertObjectAnnotation(A, 'rectangle', [67 121 43 43], 'face');
falsest ='BeerBottle';
n = size(BBOX, 1);
str_n = num2str(n);
str = strcat('Detected Faces :  ', str_n);
figure, imshow(B), title(str);

disp(str);
sizeB = size(BBOX);
sizeS = sizeB(1);

Cx = [1 sizeS];
for i = 1:sizeS
    Cx(i) = BBOX(i, 1);
end

Cs = sort(Cx);


for i = 1:sizeS
    for j = 1:sizeS
        if Cs(i) == BBOX(j,1);
        croppedImage = imcrop(A, [ BBOX(j,1), BBOX(j,2), BBOX(j,3), BBOX(j,4) ] );
        %classify
        picture = imresize(croppedImage, [227 227]);
        label = classify(myNet3, picture);
        if label == falsest;
        else
        label = string('%')+string(100*max(myNet3.predict(picture)))+string(' ')+string(label);
        
        figure, imshow(croppedImage), title(['\fontsize{17}',(label)]);
        end
        end
    end
end
end
