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

%%%% PART B %%%%

