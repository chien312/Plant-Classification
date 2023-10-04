%read the folder

dataset = imageDatastore("plant-seedlings-classification/train/", "IncludeSubfolders", true, "LabelSource", "foldernames", "FileExtensions", ".png");
dataNum = 4750;
testset = imageDatastore("plant-seedlings-classification/test/", "IncludeSubfolders", true, "LabelSource", "foldernames", "FileExtensions", ".png");
testNum = 794;

%split dataset

[dataset_t, dataset_v] = splitEachLabel(dataset, 0.5, "randomized") ;%平分0.5
%data split to train & validation
trainNum = numel(dataset_t.Files);
validateNum = numel(dataset_v.Files);

%feature extraction

feature_t = zeros(trainNum, 256*256);
for i = 1:trainNum
    [img, info] = readimage(dataset_t,i);
    img = rgb2gray(img);
    img = imresize(img, [256 256]);
    img = reshape(img, 1,[]);
    feature_t(i,:) = img;
end

feature_v = zeros(validateNum, 256*256);
for i = 1:validateNum
    img = readimage(dataset_v,i);
    img = rgb2gray(img);
    img = imresize(img, [256 256]);
    img = reshape(img, 1, []);
    feature_t(i,:) = img;
end

% nn search
idx = knnsearch(feature_t, feature_v);

%calculate accuracy

matchNum = min(trainNum, validateNum); %not equal
goodCount = 0;
for i = 1:matchNum
    if(dataset_t.Labels(idx(i)) == dataset_v.Labels(i))
        goodCount = goodCount + 1;
    end
end

accuracy = goodCount / matchNum;
disp("accuracy :"+accuracy);

%==========================Below For Tset Set==============================

%feature extraction

feature_all = zeros(dataNum, 256*256);
for i = 1:dataNum
    img = readimage(dataset,i);
    img = rgb2gray(img);
    img = imresize(img, [256 256]);
    img = reshape(img, 1,[]);
    feature_all(i,:) = img;
end

feature_test = zeros(testNum, 256*256);
for i = 1:testNum
    img = readimage(testset,i);
    img = rgb2gray(img);
    img = imresize(img, [256 256]);
    img = reshape(img, 1,[]);
    feature_test(i,:) = img;
end

% nn search
idx = knnsearch(feature_all, feature_test);

%export .csv
result = strings(testNum, 2);
result(1, 1) = "file";
result(1, 2) = "species";
for i=1:testNum
    [img,info] = readimage(testset, i);
    [filepath,name,ext] = fileparts(info.Filename);
    fileName = string(name) + string(ext);
    result(i+1, 1) = fileName;
    result(i+1, 2) = dataSet.Labels(idx(i));
end
writematrix(result, "submission.csv");