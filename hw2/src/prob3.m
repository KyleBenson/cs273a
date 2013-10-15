function [] = prob3()

  nFeatures = 4;
  
  iris=load('data/iris.txt'); y=iris(:,end); X=iris(:,1:nFeatures);
  [X y] = shuffleData(X,y); % shuffle data randomly to avoid pathological orders
  [Xtr Xte Ytr Yte] = splitData(X,y, .75); % split data into 75/25 train/test

  %split by classes
  X0 = X(y==0.0,:);
  X1 = X(y==1.0,:);
  X2 = X(y==2.0,:);

meanX0 = mean(X0);
meanX1 = mean(X1);
meanX2 = mean(X2);
covX0 = cov(X0);
covX1 = cov(X1);
covX2 = cov(X2);

if nFeatures < 3
hold on;

scatter(X0(:,1), X0(:,nFeatures), 'red');
scatter(X1(:,1), X1(:,nFeatures), 'blue');
scatter(X2(:,1), X2(:,nFeatures), 'green');
plotGauss2D(meanX0, covX0, 'red');
plotGauss2D(meanX1, covX1, 'blue');
plotGauss2D(meanX2, covX2, 'green');

saveas(gcf, '../figs/prob3a', 'pdf');
hold off;
end

% visualize classifier boundaries
bc = gaussBayesClassify( Xtr(:,1:nFeatures), Ytr );

if nFeatures < 3
plotClassify2D(bc, Xtr(:,1:nFeatures), Ytr);
saveas(gcf, '../figs/prob3b', 'pdf');
end

% compute error rates
YtrHat = predict(bc, Xtr(:,1:nFeatures));
YteHat = predict(bc, Xte(:,1:nFeatures));

trainingError = (size(YtrHat,1) - sum(YtrHat == Ytr)) / size(YtrHat,1)
testError = (size(YteHat,1) - sum(YteHat == Yte)) / size(YteHat,1)
