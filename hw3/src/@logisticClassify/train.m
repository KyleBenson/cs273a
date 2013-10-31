function obj = train(obj, Xtrain, Ytrain, stepsize, maxSteps)
% obj = train(obj, Xtrain, Ytrain, stepsize, maxSteps)
%     Xtrain = [n x d] training data features (constant feature not included)
%     Ytrain = [n x 1] training data classes (binary, e.g., +1 or -1)
%     stepsize = step size for gradient descent ("learning rate")
%     maxSteps = maximum number of steps before stopping
%
alpha = 0;
losstol = 0.000001;

if (nargin < 5) maxSteps = 5000;  end;  % max number of iterations
if (nargin < 4) stepsize = .01;   end;  % gradient descent step size

plotFlag = 1;                    % with plotting

[n,d] = size(Xtrain);            % d = dimension of data; n =
                                 % number of training data

Xtrain1= [ones(n,1), Xtrain];    % make a version of training data with the constant feature

% Get class id values and replace with values 1..C  
[Ytrain, obj.classes] = toIndex(Ytrain);
if (length(obj.classes)~=2) error('Y values must be binary!'); end;  % check correct binary labeling
Ytrain = Ytrain-max(Ytrain) + 1;             % convert Y to 0/1 for ease

obj.theta = randn(1,d+1);          % initialize weights randomly

% Outer loop of stochastic gradient descent:
iter=1;                          % iteration #
done=0;                          % end of loop flag
err=zeros(1,maxSteps);           % misclassification rate values
nlll=zeros(1,maxSteps);           % misclassification rate values
prev_loss=Inf;                            % keep track of previous
                                        % error and quit when it
                                        % goes down by a tiny amount
while (~done) 
  % Step size evolution
  %stepi = stepsize/iter;              % logistic method:
  %decreasing harmonically
  stepi = 1/iter;              % logistic method: decreasing harmonically

  % Stochastic gradient update (one pass)
  for i=1:n,  % for each data example,
    resp = (1+exp(-Xtrain1(i,:)*obj.theta')).^(-1);                       % compute logistic response
    yhati = round(resp);                                 % and prediction for Xi

    % compute gradient of regularized logistic negative log
    % likelihood loss function
    logit_term = (1+exp(-Xtrain1*obj.theta')).^(-1);
    sum_term = (logit_term - Ytrain)' * Xtrain1;
    grad = alpha + sum_term;

    %(yhati - Ytrain(i))*Xtrain1(i,:);            % Gradient-like perceptron update rule
    obj.theta = obj.theta - stepi * grad                   % Take a step down the gradient
  end;

  % Compute current error values (missclassification rate)
  err(iter)  = mean( (Ytrain~=round((1+exp(-Xtrain1*obj.theta')).^(-1))));
  % Compute regularized logistic negative log likelihood loss
log(exp(-Xtrain1*obj.theta').^(-1))
  nlll_first_term = -Ytrain'*log((1+exp(-Xtrain1*obj.theta')).^(-1));
  nlll_second_term = -(1-Ytrain')*log(1-(1+exp(-Xtrain1* ...
                                              obj.theta')).^(-1));
  alpha_term = alpha*sum(obj.theta.^2);
  nlll(iter) = n^(-1)*(nlll_first_term + nlll_second_term) + alpha_term;

  % Make plots, if desired
  if (plotFlag),
  fig(1);
  semilogx(           1:iter, nlll(1:iter), 'r-', 1:iter, err(1:iter),'g-'); %plot regularized logistic negative log likelihood loss

  fig(2); switch d,                              % Plots to help with visualization
      case 1, plot1DLinear(obj,Xtrain,Ytrain);      %  for 1D data we can display the data and the function
      case 2, plot2DLinear(obj,Xtrain,Ytrain);      %  for 2D data, just the data and decision boundary
      otherwise, % no plot for higher dimensions... %  higher dimensions visualization is hard
    end; 
  drawnow;
  end;

  %stop when no errors or out of time or changes by small amount
  done = (iter >= maxSteps || (prev_loss - nlll(iter)) < losstol);
  prev_loss = nlll(iter);
  iter = iter + 1;
end;

nlll