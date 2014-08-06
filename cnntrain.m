function [losses, modelparas] = cnntrain(opts, arch, modelparas, train_x, train_y)
%CNNTRAIN 
%   

training_size = size(train_x,3);
assert(mod(training_size, opts.batchsize)==0, 'numbatches not integer');
numbatches = training_size/opts.batchsize;

weights_inc_last.hid2out_weights_inc = zeros(size(modelparas.hid2out_weights));
weights_inc_last.featvec2hid_weights_inc = zeros(size(modelparas.featvec2hid_weights));
weights_inc_last.filters2_inc = cell(arch.numfilters1, 1);
for i = 1:arch.numfilters1
    weights_inc_last.filters2_inc{i} = zeros(size(modelparas.filters2{i}));
end
weights_inc_last.filters1_inc = zeros(size(modelparas.filters1));

losses = zeros(numbatches*opts.numepochs, 1);
for i = 1:opts.numepochs
    tic;
    randinds = randperm(training_size);
    for j = 1:numbatches
        batch_x = train_x(:,:,randinds((j-1)*opts.batchsize+1:j*opts.batchsize));
        batch_y = train_y(:, randinds((j-1)*opts.batchsize+1:j*opts.batchsize));
        
        [netstates, CE_loss] = cnnff(modelparas, arch, batch_x, batch_y);

        losses((i-1)*numbatches + j) = CE_loss;
        fprintf('Epoch %d/%d, batch %d, ce loss %f\n',i, opts.numepochs, j, CE_loss);
       
        grads = cnnbp(arch, modelparas, netstates, batch_x, batch_y);
        modelparas = cnnapplygrads(opts, arch, modelparas, grads, weights_inc_last);
    end
    toc;
end

