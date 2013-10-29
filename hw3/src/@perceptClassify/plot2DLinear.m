function plot2DLinear(obj, Xtrain, Ytrain)
% plot2DLinear(obj, Xtrain,Ytrain)
%   plot a linear classifier (data and decision boundary) when features Xtrain are 2-dim
%   wts are 1x3,  wts(0)+wts(2)*X(1)+wts(3)*X(2)
%
  [n,d] = size(Xtrain);
  if (d~=2) error('Sorry -- plot2DLinear only works on 2D data...'); end;

  wts = obj.wts;  % parameters of the linear classifier:
  
  %yhat = sign(  wts(1) + x1 * wts(2) + x2 * wts(3) )

  % Plot each class in a different color
  hold on;
  scatter(X(Y==0,1), X(Y==0,2));
  scatter(X(Y==1,1), X(Y==1,2));

  %   along with the linear decision boundary of the predictor

  hold off;
  close

  drawnow;  % ensures plot is updated immediately 

