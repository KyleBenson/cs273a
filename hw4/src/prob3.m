problem = 'd'

rand('state',0); randn('state',0);
data = load('data/spambase.data');
[N,D] = size(data);
X = data(:,1:end-1); Y=data(:,end);
[X,Y] = shuffleData(X,Y);
[Xt,Xv,Yt,Yv] = splitData(X,Y,.6);
Nt = size(Yt, 1)
Nv = size(Yv, 1)
[Xt,S] = rescale(Xt); Xv = rescale(Xv,S);

if problem == 'a'
tree = treeClassify(Xt, Yt, 0, inf, .001, inf);

YtHat = predict(tree, Xt);
YvHat = predict(tree, Xv);

trainingError = (size(YtHat,1) - sum(YtHat == Yt)) / size(YtHat,1)
testError = (size(YvHat,1) - sum(YvHat == Yv)) / size(YvHat,1)

elseif problem == 'b'
    trainingErrors = zeros(size(data, 2), 1);
    testErrors = zeros(size(data, 2), 1);
    
    depths = 1:size(data, 2);
    for depth = depths
        tree = treeClassify(Xt, Yt, 0, depth, .001, inf);

        % compute error rates
        YtHat = predict(tree, Xt);
        YvHat = predict(tree, Xv);

        trainingErrors(depth) = (size(YtHat,1) - sum(YtHat == Yt)) / size(YtHat,1);
        testErrors(depth) = (size(YvHat,1) - sum(YvHat == Yv)) / size(YvHat,1);
    end;

    hold on;
    plot(depths, trainingErrors, 'b', 'linewidth', 3);
    plot(depths, testErrors, 'g', 'linewidth', 3);
    legend('training data', 'test data');
    xlabel('depth');
    ylabel('error');
    hold off;

elseif problem == 'c'
    trainingErrors = zeros(size(data, 2), 1);
    testErrors = zeros(size(data, 2), 1);

    maxDepth = 100;
    depths = 1:maxDepth;
    trainingErrors = zeros(size(depths));
    testErrors = zeros(size(depths));

    %we collect the sum of all predictions to average them later to
    %take majority vote
    YtHat = zeros(maxDepth,Nt);
    YvHat = zeros(maxDepth,Nv);

    for depth = depths    

        for itr = 1:depth
            %bootstrap data
            [XtData YtData] = bootstrapData(Xt, Yt, Nt);

            tree = treeClassify(XtData, YtData, 0, depth, .001, inf);

            %run each classifier, summing up the predictions for an
            %average later
            YtHat(depth,:) = YtHat(depth,:) + predict(tree, Xt)'./depth;
            YvHat(depth,:) = YvHat(depth,:) + predict(tree, Xv)'./depth;
            
        end;

    end;

    %majority vote
    tmajority = sign(YtHat);
    vmajority = sign(YvHat);

    trainingErrors = (Nt - sum(YtHat == repmat(Yt', maxDepth, 1), 2)) ./ Nt;
    testErrors = (Nv - sum(YvHat == repmat(Yv', maxDepth, 1), 2)) ./ Nv;

    %plot errors vs. depth    

    hold on;
    plot(depths, trainingErrors, 'b', 'linewidth', 3);
    plot(depths, testErrors, 'g', 'linewidth', 3);
    legend('training data', 'test data');
    xlabel('# learners');
    ylabel('error');
    hold off;

elseif problem == 'd'
    maxDepth = 25;

    %we collect the sum of all predictions to average them later to
    %take majority vote
    YtHat = zeros(maxDepth,Nt);
    YvHat = zeros(maxDepth,Nv);

    depth = maxDepth;
    nitr = depth;
    nSamplesToTest = floor(logspace(log10(round(0.1*Nt)), log10(Nt)));
    for nSamples = nSamplesToTest    

        for itr = 1:nitr
            %bootstrap data
            [XtData YtData] = bootstrapData(Xt, Yt, nSamples);

            tree = treeClassify(XtData, YtData, 0, depth, .001, inf);

            %run each classifier, summing up the predictions for an
            %average later
            YtHat(depth,:) = YtHat(depth,:) + predict(tree, Xt)';
            YvHat(depth,:) = YvHat(depth,:) + predict(tree, Xv)';
            
        end;

    end;

    %majority vote
    tmajority = sign(YtHat./depth);
    vmajority = sign(YvHat./depth);

    trainingErrors = (Nt - sum(YtHat == repmat(Yt', maxDepth, 1), 2)) ./ Nt;
    testErrors = (Nv - sum(YvHat == repmat(Yv', maxDepth, 1), 2)) ./ Nv;

    %plot errors vs. nSamplesToTest    

    hold on;
    plot(nSamplesToTest, trainingErrors, 'b', 'linewidth', 3);
    plot(nSamplesToTest, testErrors, 'g', 'linewidth', 3);
    legend('training data', 'test data');
    xlabel('# samples in bootstrap');
    ylabel('error');
    hold off;

end;
