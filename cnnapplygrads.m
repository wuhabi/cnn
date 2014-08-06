function [ modelparas ] = cnnapplygrads(opts, arch, modelparas, grads, weights_inc_last)
%CNNAPPLYGRAD 
% 

weight_decay = 0.0005;

%%
% update hid2out_weights
hid2out_weights_inc = opts.momentum*weights_inc_last.hid2out_weights_inc - opts.learningrate*grads.hid2out_weights_grad;
weights_inc_last.hid2out_weights_inc = hid2out_weights_inc;
modelparas.hid2out_weights = modelparas.hid2out_weights + hid2out_weights_inc;

% update out_bias
out_bias_inc = -opts.learningrate*grads.out_bias_grad;
modelparas.out_bias = modelparas.out_bias + out_bias_inc;

% update featvec2hid_weights
featvec2hid_weights_inc = opts.momentum*weights_inc_last.featvec2hid_weights_inc - opts.learningrate*grads.featvec2hid_weights_grad;
weights_inc_last.featvec2hid_weights_inc = featvec2hid_weights_inc;
modelparas.featvec2hid_weights = modelparas.featvec2hid_weights + featvec2hid_weights_inc;

% update filters2
filters2_inc = cell(arch.numfilters1,1);
for i = 1:arch.numfilters1
    filters2_inc{i} = opts.momentum*weights_inc_last.filters2_inc{i} - opts.learningrate*grads.filters2_grad{i};
    weights_inc_last.filters2_inc{i} = filters2_inc{i};
    modelparas.filters2{i} = modelparas.filters2{i} + filters2_inc{i};
end

% update filters2_bias
filters2_bias_inc = -opts.learningrate*grads.filters2_bias_grad;
modelparas.filters2_bias = modelparas.filters2_bias + filters2_bias_inc;

% update filters1
filters1_inc = opts.momentum*weights_inc_last.filters1_inc - opts.learningrate*grads.filters1_grad;
weights_inc_last.filters1_inc = filters1_inc;
modelparas.filters1 = modelparas.filters1 + filters1_inc;

% update filters1_bias
filters1_bias_inc = -opts.learningrate*grads.filters1_bias_grad;
modelparas.filters1_bias = modelparas.filters1_bias + filters1_bias_inc;

end

