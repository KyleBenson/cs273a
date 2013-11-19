mnist=load('data/mnist.txt'); % load the text file
Y=mnist(:,1); X = mnist(:,2:end); % note different order of data
keep = (Y==0 | Y==8); % create binary classification problem between zeros and eights
Y=Y(keep)/8; X=X(keep,:);

data_point = 5;
imagesc( reshape(X(data_point,:), [28 28])' ); colormap(1-gray(256));

%split into validation and training data
[X Y] = shuffleData(X,Y); % shuffle data randomly to avoid pathological orders
[Xtr Xte Ytr Yte] = splitData(X,Y, .75); % split data into 75/25 train/test

[XtrS M S] = rescale(Xtr);
lc = logisticClassify(XtrS, Ytr, 0.1, 1000);
training_error = err(lc, Xtr, Ytr)
test_error = err(lc, Xte, Yte)

% part b

nFeat = 2;

[XtP P] = fproject(Xtr, nFeat);
[XtP M S] = rescale(XtP);
XvP = rescale( fproject(Xte,nFeat,P), M,S); % apply to validation data
lc = logisticClassify(XtrS, Ytr, 0.01, 5000);

training_error_2features = err(lc, Xtr, Ytr)
test_error_2features = err(lc, Xte, Yte)


nFeat = 10;

[XtP P] = fproject(Xtr, nFeat);
[XtP M S] = rescale(XtP);
XvP = rescale( fproject(Xte,nFeat,P), M,S); % apply to validation data
lc = logisticClassify(XtrS, Ytr, 0.01, 5000);

training_error_10features = err(lc, Xtr, Ytr)
test_error_10features = err(lc, Xte, Yte)
