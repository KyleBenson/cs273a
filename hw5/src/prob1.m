problem = 'd'

iris=load('data/iris.txt');
X = iris(:,1:2); Y=iris(:,end);
XA = X(Y<2,:); YA=Y(Y<2); % 0 vs 1
YA = (YA.*2)-1; % set to -1,1
N = size(XA, 1);

if problem == 'a'

scatter(XA(:,1), XA(:,2));

elseif problem == 'b'

K=20
[assign, clusters, sumd] = kmeans(XA, K);

elseif problem == 'c'

K=20
method='min'; %single-link
%method='max'; %complete-link
[assign, join] = agglomCluster(XA, K, method);

elseif problem == 'd'

K=5
[assign, clusters, R, llh] = emCluster(XA, K);

end;