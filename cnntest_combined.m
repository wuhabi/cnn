function [mean_err, max_err, combined_err] = cnntest_combined( arch, modelparas1, modelparas2, test_x, test_y )
%CNNTEST_COMBINED 
%   
arch.poolstyle = 'mean';
arch.mode = 'test';
[netstates1, ~] = cnnff(modelparas1, arch, test_x, test_y);
probs1 = netstates1.outstates;
[~, predict] = max(probs1);
[~, target] = max(test_y);
bad = find(predict ~= target);
mean_err = numel(bad) / size(test_x, 3);

arch.poolstyle = 'stoc_max';
arch.mode = 'test';
[netstates2, ~] = cnnff(modelparas2, arch, test_x, test_y);
probs2 = netstates2.outstates;
[~, predict] = max(probs2);
[~, target] = max(test_y);
bad = find(predict ~= target);
max_err = numel(bad) / size(test_x, 3);

probs3 = 0.5*(probs1 + probs2);
[~, predict] = max(probs3);
[~, target] = max(test_y);
bad = find(predict ~= target);
combined_err = numel(bad) / size(test_x, 3);

end

