problem = 'b'

iris=load('data/iris.txt');
X = iris(:,1:2); Y=iris(:,end);
XA = X(Y<2,:); YA=Y(Y<2); % 0 vs 1
YA = (YA.*2)-1; % set to -1,1
N = size(XA, 1);

if problem == 'a'

dim = size(XA,2) + 1; % size of theta vector (including b)
H = ones(dim,1)*2;
H(1) = 0;
H = diag(H);

A = bsxfun(@times, XA, YA);
A = -1*[YA A];
b = -1*ones(N,1);

theta = quadprog(H,zeros(dim,1),A,b,[],[],-100000*ones(dim,1),100000*ones(dim,1));

% separate parameters
%b = theta(1);
%theta = theta(2:end);

pc = perceptClassify(XA, YA);
pc = setWeights(pc, theta');
plotClassify2D(pc, XA, YA);

elseif problem == 'b'

dim = N; % size of theta vector (alphas)
Ysquare = YA * YA';
Xsquare = XA * XA';
H = Ysquare.*Xsquare;
f = -1*ones(N,1);

%Aeq = diag(YA);
Aeq = YA';
%beq = zeros(dim,1);
beq = [0];

theta = quadprog(H,f,[],[],Aeq, beq,zeros(N,1),100000*ones(N,1))

% theta is a N-dim vector of alpha weights,
% so we need to extract the actual theta parameters from it
%TODO

pc = perceptClassify(XA, YA);
pc = setWeights(pc, [0 theta']');
plotClassify2D(pc, XA, YA);

end;