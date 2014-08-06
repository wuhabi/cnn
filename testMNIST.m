% imwuhaibing@gmail.com  2014/06/29
% 

clear; clc;

% Load dataset
train_x = loadMNISTImages('./data/train-images-idx3-ubyte');
train_x = reshape(train_x,28,28,[]);
train_y = loadMNISTLabels('./data/train-labels-idx1-ubyte');
train_y(train_y==0) = 10; % Remap 0 to 10
transmat = eye(10,10);
train_y = transmat(:,train_y);

test_x = loadMNISTImages('./data/t10k-images-idx3-ubyte');
test_x = reshape(test_x,28,28,[]);
test_y = loadMNISTLabels('./data/t10k-labels-idx1-ubyte');
test_y(test_y==0) = 10; % Remap 0 to 10
test_y = transmat(:,test_y);

fprintf('Load data successfully!\n');

DEBUG = false;
if DEBUG
    arch.inputdim = 28;
    arch.filterdim1 = 5; arch.numfilters1 = 2; arch.poolscale1 = 2;
    arch.filterdim2 = 5; arch.numfilters2 = 2; arch.poolscale2 = 2;
    arch.poolstyle = 'stoc_mean';
    arch.acttype = 'sigm';
    arch.hiddim = 12;
    arch.outputdim = 10; 
    modelparas = cnninit(arch);
    db_train_x = train_x(:,:,1:10);
    db_train_y = train_y(:,1:10);
    
    numgrads = computenumgrad(arch, modelparas, db_train_x, db_train_y);
    [netstates, ~] = cnnff(modelparas, arch, db_train_x, db_train_y);
    grads = cnnbp(arch, modelparas, netstates, db_train_x, db_train_y);
else
    arch.inputdim = 28;
    arch.filterdim1 = 5; arch.numfilters1 = 20; arch.poolscale1 = 2;
    arch.filterdim2 = 5; arch.numfilters2 = 40; arch.poolscale2 = 2;
    arch.poolstyle = 'max'; % max, mean, stoc_mean, stoc_max
    arch.acttype = 'relu'; % sigm or relu
    arch.conv_dropout_fraction = 0.5;
    arch.pool_dropout_fraction = -1;
    arch.featvec_dropout_fraction = 0.2;
    arch.hid_dropout_fraction = 0.5;
    arch.hiddim = 1000;
    arch.outputdim = 10; 
    
    opts.numepochs = 500;
    opts.batchsize = 100;
    opts.learningrate = 0.1;
    opts.momentum = 0.9;
    
    modelparas = cnninit(arch);
    tStart = tic;
    [losses, modelparas] = cnntrain_test(opts, arch, modelparas, train_x, train_y, test_x, test_y);
    tElapsed = toc(tStart);
    figure; plot(losses);
    
%     err = cnntest(arch, modelparas, test_x, test_y);
%     fprintf('test error rate is %f\n',err);
%     
%     train_err1 = cnntest(arch, modelparas, train_x(:,:,1:20000), train_y(:,1:20000));
%     train_err2 = cnntest(arch, modelparas, train_x(:,:,20001:40000), train_y(:,20001:40000));
%     train_err3 = cnntest(arch, modelparas, train_x(:,:,40001:60000), train_y(:,40001:60000));
%     fprintf('train error rate is %f\n',(train_err1+train_err2+train_err3)/3);
%     
%     fileID = fopen('./results/result.txt','a');
%     fprintf(fileID, '======================================================\n');
%     fprintf(fileID, 'numfilters1: %d\n',arch.numfilters1);
%     fprintf(fileID, 'numfilters2: %d\n',arch.numfilters2);
%     fprintf(fileID, 'pooling style: %s\n',arch.poolstyle);
%     fprintf(fileID, 'activation type: %s\n',arch.acttype);
%     fprintf(fileID, 'number of epochs: %d\n',opts.numepochs);
%     fprintf(fileID, 'learning rate: %f\n',opts.learningrate);
%     fprintf(fileID, 'momentum: %f\n',opts.momentum);
%     fprintf(fileID, 'test error: %f\n',err);
%     fprintf(fileID, 'train error: %f\n',(train_err1+train_err2+train_err3)/3);
%     fprintf(fileID, 'training time: %f\n',tElapsed);
%     fclose(fileID);
end
