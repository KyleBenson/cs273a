problem = 'f'

X = load('data/faces.txt'); % load face dataset

% subtract mean to make data zero-mean
mu = mean(X);
X = X - repmat(mu, size(X,1), 1);

[U, S, V] = svds(X, 500);
W = U*S;
sizew = size(W)
sizev = size(V)

if problem == 'c'

maxK = 10;
dataPoints = ones(maxK,1);
Krange = 1:1:maxK;
for K = Krange
    Xhat = W(:,1:K)*V(:,1:K)';
    thisPointMse = mean(mean((X-Xhat).^2));
    dataPoints(K) = thisPointMse;
end;

hold on;
plot(dataPoints);
ylabel('MSE');
xlabel('K = 1:10');
hold off;

elseif problem == 'd'

for j = 1:5
alpha = 2*median(abs(W(:,j)));

img1 = mu + alpha*V(:,j)';
img2 = mu - alpha*V(:,j)';

img1 = reshape(img1,[24 24]);
img2 = reshape(img2,[24 24]);
imagesc(img1); axis square; colormap gray;
saveas(gcf, ['../figs/prob3d_plus' int2str(j)], 'pdf');
imagesc(img2); axis square; colormap gray;
saveas(gcf, ['../figs/prob3d_minus' int2str(j)], 'pdf');

end;

elseif problem == 'e'

for idx = 1:100:2500
    % pick some data at random or otherwise
    figure; hold on; axis ij; colormap(gray);
    range = max(W(idx,1:2)) - min(W(idx,1:2)); % find range of
                                               % coordinates to be
                                               % plotted
    scale = [200 200]./range;
    % want 24x24 to be visible but not large on new scale
    for i=idx, imagesc(W(i,1)*scale(1),W(i,2)*scale(2), reshape(X(i,:),24,24));
    end;
    pause(2);
end;

elseif problem == 'f'

for K = [5 10 50 500]
    for faceIdx = [42 620]
        img = W(faceIdx,1:K)*V(:,1:K)';
        img = reshape(img,[24 24]);
        imagesc(img); axis square; colormap gray;
        title(['Face ' int2str(faceIdx) ' Principal Directions 1:' int2str(K)]);
        %saveas(gcf, ['../figs/prob3f_face' int2str(faceIdx) '_K' int2str(K)], 'pdf');
        %pause(2);
    end;
end;

elseif problem == 'scratch'
i = 3;
img = reshape(X(i,:),[24 24]); % convert vectorized datum to 24x24
                               % image patch
imagesc(img); axis square; colormap gray;
% display an image patch; you may have to squint

end;