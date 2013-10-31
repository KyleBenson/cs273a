function plot2DLinear(obj, Xtrain, Ytrain)
% plot2DLinear(obj, Xtrain,Ytrain)
%   plot a linear classifier (data and decision boundary) when features Xtrain are 2-dim
%   wts are 1x3,  wts(0)+wts(2)*X(1)+wts(3)*X(2)
%
  [n,d] = size(Xtrain);
  if (d~=2) error('Sorry -- plot2DLinear only works on 2D data...'); end;

  wts = obj.theta;  % parameters of the linear classifier:
  
  %yhat = sign(  wts(1) + x1 * wts(2) + x2 * wts(3) )

  % Plot each class in a different color
  clf;
  hold on;
  i0 = find(Ytrain==0); i1 = find(Ytrain==1);
  scatter(Xtrain(i0,1), Xtrain(i0,2));
  scatter(Xtrain(i1,1), Xtrain(i1,2));

  %   along with the linear decision boundary of the predictor
  xs = linspace(min(Xtrain(:,1)), max(Xtrain(:,1)), 200);
  ys = (xs.*wts(2)+wts(1))./(-wts(3));
  plot(xs, ys);

  hold off;

  drawnow;  % ensures plot is updated immediately 
