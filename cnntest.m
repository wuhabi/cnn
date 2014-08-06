function [ err ] = cnntest( arch, modelparas, test_x, test_y )
%CNNTEST 
%   

[netstates, ~] = cnnff(modelparas, arch, test_x, test_y);

probs = netstates.outstates;

[~, predict] = max(probs);
[~, target] = max(test_y);

bad = find(predict ~= target);
err = numel(bad) / size(test_x, 3);

end

