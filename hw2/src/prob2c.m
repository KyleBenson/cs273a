function [] = prob2c()
  mcycle=load('data/mcycle80.txt'); y=mcycle(:,end); X=mcycle(:,1:end-1);
[X y] = shuffleData(X,y); % shuffle data randomly to avoid pathological orders
 
K=[1, 2, 4, 8, 16, 32, 64];

for k=1:length(K)
	nFolds = 5;
knnMse(k) = 0;
for iFold = 1:nFolds
	      [Xti,Xvi,Yti,Yvi] = crossValidate(X,y,nFolds,iFold);  
knn = knnRegress(Xti, Yti, k); % replace or set K to some integer

YviHat = predict( knn, Xvi ); % make predictions on Xtest
knnMse(k) = knnMse(k) + mean((Yvi-YviHat).^2);
end;

knnMse(k) = knnMse(k) ./ nFolds;

end;

figure;
semilogx(K, knnMse, 'red');
saveas(gcf, '../figs/prob2c', 'pdf');
