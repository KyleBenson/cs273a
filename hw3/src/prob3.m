% X1 = [0.05 0.24 0.28 0.53 0.61 0.48 0.58 0.76 0.80 0.95]';
% Y1 = [ 0 0 0 0 0 1 1 1 1 1]';

%iris=load('~/repos/cs273a/hw3/src/data/iris.txt');
iris=load('data/iris.txt');
X = iris(:,1:2); Y=iris(:,end);
XA = X(Y<2,:); YA=Y(Y<2); % 0 vs 1
XB = X(Y>0,:); YB=Y(Y>0); % 1 vs 2

%lc = logisticClassify(X1,Y1, .1, 1000);
%lc = logisticClassify(XB,YB, 0.01, 5000);
%lc = logisticClassify(XA,YA, 0.10, 1000);

% part g

degree = 3;
[Xsh Ysh] = shuffleData(XB,YB); % shuffle data randomly to avoid pathological orders
XshP = fpoly(Xsh, degree, false); % create polynomial features up
                                  % to given degree, no "1" feature
[XshP, M,S] = rescale(XshP);
[Xtr Xte Ytr Yte] = splitData(XshP, Ysh, .75); % split data into 75/25 train/test

lc = logisticClassify(Xtr,Ytr, 0.1, 1000);

Yte = Yte-max(Yte) + 1;
Ytr = Ytr-max(Ytr) + 1;

training_error = err(lc, Xtr, Ytr)
test_error = err(lc, Xte, Yte)

% part h

transform = @(x) rescale( fpoly(x,degree,false), M,S);

YB = YB - max(YB) + 1;
plotClassify2D( lc, XB,YB, transform);
