function [] = prob2b()
  mcycle=load('data/mcycle80.txt'); y=mcycle(:,end); X=mcycle(:,1:end-1);
[X y] = shuffleData(X,y); % shuffle data randomly to avoid pathological orders
[Xtr Xte Ytr Yte] = splitData(X,y, .75); % split data into 75/25 train/test
 
K=[1, 2, 4, 8, 16, 32, 64];

for k=1:length(K)
	knn = knnRegress(Xtr, Ytr, k); % replace or set K to some integer
	YtrHat = predict( knn, Xtr ); % make predictions on Xtrain
	YteHat = predict( knn, Xte ); % make predictions on Xtest
	knnMseTr(k) = mean((Ytr-YtrHat).^2);
	knnMseTe(k) = mean((Yte-YteHat).^2);
end;

figure;
semilogx(K, knnMseTr, 'red', K, knnMseTe, 'green');
saveas(gcf, '../plots/prob2b', 'pdf');
