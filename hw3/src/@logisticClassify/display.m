function display(obj)
% display(learner) : display information on the perceptron (its coefficients)
  fprintf('Linear Binary Classifier; %d features\n',length(obj.theta));
  disp(obj.theta);

