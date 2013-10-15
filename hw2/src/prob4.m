function [] = prob4()

  numFeatures = 2;

  iris=load('data/iris.txt'); y=iris(:,end); X=iris(:,1:numFeatures);
  [X y] = shuffleData(X,y); % shuffle data randomly to avoid pathological orders
  [Xtr Xte Ytr Yte] = splitData(X,y, .75); % split data into 75/25 train/test

  for depth = 1:size(iris, 2)
tree = treeClassify(Xtr, Ytr, 0, depth, -1, inf);

if numFeatures < 3
plotClassify2D(tree, Xtr, Ytr);
saveas(gcf, '../figs/prob4', 'pdf');
end;

% compute error rates
YteHat = predict(tree, Xte);

depth
testError = (size(YteHat,1) - sum(YteHat == Yte)) / size(YteHat,1)

  end;
