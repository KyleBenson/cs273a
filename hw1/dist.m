function D = mydist(x, X)
% Calculates the vector of Euclidean distances between a single vector x
% and a collection of data points (stored in a matrix) X.

differences = bsxfun(@minus, x, X);
squared_differences = differences.^2;
sum_squared_differences = sum(squared_differences, 2);
D = sqrt(sum_squared_differences);

end
