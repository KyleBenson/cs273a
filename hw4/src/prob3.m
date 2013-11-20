problem = 'd'

%rand('state',0); randn('state',0);
data = load('data/spambase.data');
[N,D] = size(data);
X = data(:,1:end-1); Y=data(:,end);
[X,Y] = shuffleData(X,Y);
[Xt,Xv,Yt,Yv] = splitData(X,Y,.6);
Yt(Yt == 0) = -1;
Yv(Yv == 0) = -1;
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
    maxDepth = 15;
    %maxDepth = 100;
    depths = round(logspace(0,log10(maxDepth)));
    maxDepth = size(depths, 2); % so that we create prediction
                                % vectors of the right size
    
    %we collect the sum of all predictions to average them later to
    %take majority vote
    YtHat = zeros(maxDepth,Nt);
    YvHat = zeros(maxDepth,Nv);

    depthItr = 1;
    for depth = depths    

        for itr = 1:depth
            %bootstrap data
            [XtData YtData] = bootstrapData(Xt, Yt, Nt);

            tree = treeClassify(XtData, YtData, 0, inf, 0, inf);

            %run each classifier, summing up the predictions for an
            %average later
            % treeClassify outputs 0/1 labels, so change to -1/+1
            tpredictions = predict(tree, Xt);
            vpredictions = predict(tree, Xv);
            YtHat(depthItr,:) = YtHat(depthItr,:) + tpredictions'./depth;
            YvHat(depthItr,:) = YvHat(depthItr,:) + vpredictions'./depth;
            
        end;
        depthItr = depthItr + 1;
    end;

    %majority vote
    tmajority = sign(YtHat);
    vmajority = sign(YvHat);

    tmajority(tmajority == 0) = 1;
    vmajority(vmajority == 0) = 1;
    
    tmajority = bsxfun(@eq, tmajority, Yt');
    vmajority = bsxfun(@eq, vmajority, Yv');
    
    tmajority = sum(tmajority, 2); % get # right for each depth
    vmajority = sum(vmajority, 2); % get # right for each depth

    trainingErrors = (Nt - tmajority) ./ Nt;
    testErrors = (Nv - vmajority) ./ Nv;

    %plot errors vs. depth    

    hold on;
    plot(depths, trainingErrors, 'b', 'linewidth', 3);
    plot(depths, testErrors, 'g', 'linewidth', 3);
    legend('training data', 'test data');
    xlabel('# learners');
    ylabel('error');
    hold off;

elseif problem == 'd'
    nitr = 25;
    testPoints = 100;
    
    %we collect the sum of all predictions to average them later to
    %take majority vote
    YtHat = zeros(testPoints,Nt);
    YvHat = zeros(testPoints,Nv);
    
    nSamplesToTest = floor(logspace(log10(round(0.1*Nt)), log10(Nt), testPoints));
    testItr = 1;
    for nSamples = nSamplesToTest    

        for itr = 1:nitr
            %bootstrap data
            [XtData YtData] = bootstrapData(Xt, Yt, nSamples);

            tree = treeClassify(XtData, YtData, 0, inf, 0, inf);

            %run each classifier, summing up the predictions for an
            %average later
            % treeClassify outputs 0/1 labels, so change to -1/+1
            tpredictions = predict(tree, Xt);
            vpredictions = predict(tree, Xv);
            YtHat(testItr,:) = YtHat(testItr,:) + tpredictions';
            YvHat(testItr,:) = YvHat(testItr,:) + vpredictions';
            
        end;
        testItr = testItr + 1;
    end;

    %majority vote
    tmajority = sign(YtHat./depth);
    vmajority = sign(YvHat./depth);

    % prefer class 1 when tied
    tmajority(tmajority == 0) = 1;
    vmajority(vmajority == 0) = 1;
    
    tmajority = bsxfun(@eq, tmajority, Yt');
    vmajority = bsxfun(@eq, vmajority, Yv');
    
    tmajority = sum(tmajority, 2); % get # right for each depth
    vmajority = sum(vmajority, 2); % get # right for each depth

    trainingErrors = (Nt - tmajority) ./ Nt;
    testErrors = (Nv - vmajority) ./ Nv;

    %plot errors vs. nSamplesToTest    

    hold on;
    plot(nSamplesToTest, trainingErrors, 'b', 'linewidth', 3);
    plot(nSamplesToTest, testErrors, 'g', 'linewidth', 3);
    legend('training data', 'test data');
    xlabel('# samples in bootstrap');
    ylabel('error');
    hold off;

end;
