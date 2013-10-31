X1 = [0.05 0.24 0.28 0.53 0.61 0.48 0.58 0.76 0.80 0.95]';
Y1 = [ 0 0 0 0 0 1 1 1 1 1]';

iris=load('~/repos/cs273a/hw3/src/data/iris.txt');
X = iris(:,1:2); Y=iris(:,end);
XA = X(Y<2,:); YA=Y(Y<2); % 0 vs 1
XB = X(Y>0,:); YB=Y(Y>0); % 1 vs 2

lc = logisticClassify(X1,Y1, .1, 1000);
%lc = logisticClassify(XB,YB, .000001, 500);
%lc = logisticClassify(XA,YA, .000001, 500);

degree = 2;

XAP = fpoly(XA, degree, false); % create polynomial features up
                                  % to given degree, no "1" feature
[XtrP, M,S] = rescale(XAP);

transform = @(x) rescale( fpoly(x,degree,false), M,S);

%plotClassify2D( lc, XA,YA, transform);

plotClassify2D( lc, XtrP,YA, transform);