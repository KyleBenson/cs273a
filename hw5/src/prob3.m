X = load('data/faces.txt'); % load face dataset
img = reshape(X(i,:),[24 24]); % convert vectorized datum to 24x24
                               % image patch
imagesc(img); axis square; colormap gray;
% display an image patch; you may have to squint

% subtract mean to make data zero-mean
mean(X)
X = X - mean(X)