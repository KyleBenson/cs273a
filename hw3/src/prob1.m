mcycle=load('data/mcycle80.txt'); y=mcycle(:,end); X=mcycle(:,1:end-1);
[X y] = shuffleData(X,y) % shuffle data randomly to avoid pathological orders
[Xtr Xte Ytr Yte] = splitData(X,y, .75); % split data into 75/25 train/test


lr = linearRegress( Xtr, Ytr );
xs = [0:.01:2]';
ys = predict( lr, xs );

% test on training data
YtrHat = predict(lr, Xtr);
trainingMSE = mean((YtrHat - Ytr).^2)

% test on test data
YteHat = predict(lr, Xte);
testMSE = mean((YteHat - Yte).^2)

% plot data and predictor
hold on;
plot(xs, ys);
scatter(Xtr, Ytr);
hold off;
%saveas(gcf, 'prob1b', 'pdf');
close(gcf);

%% done part b

% gather up MSE's
testMSEs = [];
trainingMSEs = [];

% train regression model on several dimensions
degrees=[1 3 5 7 10 18];
for degree=degrees
    XtrP = fpoly(Xtr, degree, false);
    [XtrP, M,S] = rescale(XtrP);
    XteP = rescale(Xte, M,S);
    lr = linearRegress( XtrP, Ytr );

    xs = [0:.01:2]';
    xsP = fpoly(xs, degree, false);
    ysP = predict( lr, xsP );

    % test on training data
    YtrP = predict (lr, XtrP);
    trainingMSEs = [trainingMSEs mean((YtrP - Ytr).^2)]
    % test on test data
    YteP = predict (lr, XteP);
    testMSEs = [testMSEs mean((YteP - Yte).^2)]

    % plot the learned prediction functions
    hold on;
    plot(xs, ysP);
    scatter(Xtr, YtrP);
    hold off;
    saveas(gcf, ['../figs/prob1c_deg' int2str(degree)], 'pdf');
    close(gcf);
end;


% now we plot the MSEs as a function of the degree
semilogy(degrees, trainingMSEs, degrees, testMSEs);
xlabel('degree');
ylabel('MSE');
legend('training data', 'test data');
saveas(gcf, '../figs/prob1c_mse', 'pdf');

% part c
% vary alpha with degree 18
degree = 18;
testMSEs = [];
trainingMSEs = [];
alphas=logspace(-6,1,15);

for alpha=alphas
    XtrP = fpoly(Xtr, degree, false);
    [XtrP, M,S] = rescale(XtrP);
    XteP = rescale(Xte, M,S);
    lr = linearRegress( XtrP, Ytr, alpha );

    xs = [0:.01:2]';
    xsP = fpoly(xs, degree, false);
    ysP = predict( lr, xsP );

    % test on training data
    YtrP = predict (lr, XtrP);
    trainingMSEs = [trainingMSEs mean((YtrP - Ytr).^2)]
    % test on test data
    YteP = predict (lr, XteP);
    testMSEs = [testMSEs mean((YteP - Yte).^2)]

end;

% plot our MSEs for different alpha
semilogx(alphas, trainingMSEs, alphas, testMSEs);
xlabel('alpha');
ylabel('MSE');
legend('training data', 'test data');
saveas(gcf, '../figs/prob1d', 'pdf');

