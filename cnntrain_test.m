function [losses, modelparas] = cnntrain_test(opts, arch, modelparas, train_x, train_y, test_x, test_y)
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
        
        arch.poolstyle = 'max';
        arch.mode = 'train';
        [netstates, CE_loss] = cnnff(modelparas, arch, batch_x, batch_y);

        losses((i-1)*numbatches + j) = CE_loss;
        fprintf('Epoch %d/%d, batch %d, ce loss %f\n',i, opts.numepochs, j, CE_loss);
       
        grads = cnnbp(arch, modelparas, netstates, batch_x, batch_y);
        modelparas = cnnapplygrads(opts, arch, modelparas, grads, weights_inc_last);
    end
    toc;
    
    if mod(i,1)==0
        arch.poolstyle = 'max';
        arch.mode = 'test';
        test_err = cnntest(arch, modelparas, test_x, test_y);
%         if test_err<0.0055
%             save(['./models/modelparas_epoch',num2str(i),'.mat'],'modelparas');
%         end
        if i>=10 && mod(i,10)==0
            train_err1 = cnntest(arch, modelparas, train_x(:,:,1:10000), train_y(:,1:10000));
            train_err2 = cnntest(arch, modelparas, train_x(:,:,10001:20000), train_y(:,10001:20000));
            train_err3 = cnntest(arch, modelparas, train_x(:,:,20001:30000), train_y(:,20001:30000));
            train_err4 = cnntest(arch, modelparas, train_x(:,:,30001:40000), train_y(:,30001:40000));
            train_err5 = cnntest(arch, modelparas, train_x(:,:,40001:50000), train_y(:,40001:50000));
            train_err6 = cnntest(arch, modelparas, train_x(:,:,50001:60000), train_y(:,50001:60000));
            train_err = (train_err1+train_err2+train_err3+train_err4+train_err5+train_err6)/6;
            fileID = fopen('./result1.txt','a');
            fprintf(fileID, '----------------------------------------------------\n');
            fprintf(fileID, 'train error: %f\n',train_err);
            fclose(fileID);
        end
        
        fileID = fopen('./result1.txt','a');
        fprintf(fileID, '======================================================\n');
        fprintf(fileID, 'numfilters1: %d\n',arch.numfilters1);
        fprintf(fileID, 'numfilters2: %d\n',arch.numfilters2);
        fprintf(fileID, 'convolution dropout fraction: %f\n',arch.conv_dropout_fraction);
        fprintf(fileID, 'pooling dropout fraction: %f\n',arch.pool_dropout_fraction);
        fprintf(fileID, 'featvec dropout fraction: %f\n',arch.featvec_dropout_fraction);
        fprintf(fileID, 'hid dropout fraction: %f\n',arch.hid_dropout_fraction);
        fprintf(fileID, 'hiddim: %d\n',arch.hiddim);
        fprintf(fileID, 'pooling style: %s\n',arch.poolstyle);
        fprintf(fileID, 'activation type: %s\n',arch.acttype);
        fprintf(fileID, 'number of epochs: %d\n',i);
        fprintf(fileID, 'learning rate: %f\n',opts.learningrate);
        fprintf(fileID, 'momentum: %f\n',opts.momentum);
        fprintf(fileID, 'test error: %f\n',test_err);
%         fprintf(fileID, 'train error: %f\n',train_err);
        fclose(fileID);
    end
end

