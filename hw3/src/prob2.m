iris=load('~/repos/cs273a/hw3/src/data/iris.txt');
X = iris(:,1:2); Y=iris(:,end);
XA = X(Y<2,:); YA=Y(Y<2); % 0 vs 1
XB = X(Y>0,:); YB=Y(Y>0); % 1 vs 2

%%%% PART A %%%%

%hold on;
%scatter(X(Y==0,1), X(Y==0,2));
%scatter(X(Y==1,1), X(Y==1,2));
%hold off;
%%saveas(gcf, '../figs/prob2a_0v1', 'pdf');
%close
%
%hold on;
%scatter(X(Y==1,1), X(Y==1,2));
%scatter(X(Y==2,1), X(Y==2,2));
%hold off;
%%saveas(gcf, '../figs/prob2a_1v2', 'pdf');
%close

%%%% PART C %%%%
step = 0.05;
nIter = 100;

%pc = perceptClassify(XB,YB, step,nIter);

pc = perceptClassify(XA,YA, step, nIter);

%%%% PART D %%%%
% xs = XA;
% %xs = XB;
% ys = YA.*2-1;
% %ys = (YB-1).*2-1;

% lc = linearRegress(xs, ys);

% pc = perceptClassify();
% weights = getWeights(lc);
% pc = setWeights(pc, weights);
% plot2DLinear(pc, xs, ys);