function Yte = predict(obj,Xte)
% predict(obj,Xtest) : predict output classes on test data
%Yte = (sign(obj.theta(1) + Xte*obj.theta(2:end)') + 1) /2;
  Yte = round( (1+exp(-obj.theta(1) - Xte*obj.theta(2:end)')).^(-1));   % output prediction 0/1
