function Yte = predict(obj,Xte)
% predict(obj,Xtest) : predict output classes on test data

  Yte = sign( obj.wts(1) + Xte*obj.wts(2:end)');   % output prediction +/- 1
  Yte = obj.classes( Yte/2 + 1.5 );                % convert to original class values

