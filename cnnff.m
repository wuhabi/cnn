 function [ netstates, loss ] = cnnff(modelparas, arch, batch_x, batch_y)
%CNNFF 
%   

%% layer 1
batchsize = size(batch_x,3);
pooleddim1 = (arch.inputdim-arch.filterdim1+1)/arch.poolscale1;
netstates.inputmaps1 = batch_x;
netstates.ispos1 = cell(arch.numfilters1, 1);
netstates.convedmaps1 = cell(arch.numfilters1, 1);
netstates.conv1_dropout_mask = cell(arch.numfilters1, 1);
netstates.pool1_dropout_mask = cell(arch.numfilters1, 1);
netstates.pooledmaps1 = cell(arch.numfilters1, 1);
netstates.posinds1 = cell(arch.numfilters1, 1);
 
for i = 1:arch.numfilters1
    % convolution
    inputs2conv1 = convn(netstates.inputmaps1, rot90(modelparas.filters1(:,:,i),2), 'valid') + modelparas.filters1_bias(i);
    if strcmp(arch.acttype, 'sigm') % sigmoid unit
        netstates.convedmaps1{i} = 1./(1+exp(-inputs2conv1));
    elseif strcmp(arch.acttype, 'relu') % rectified linear unit
        netstates.ispos1{i} = double(inputs2conv1>0);
        netstates.convedmaps1{i} = inputs2conv1.*netstates.ispos1{i};
    end
    
    % dropout
    if arch.conv_dropout_fraction>0
        if strcmp(arch.mode, 'test')
            netstates.convedmaps1{i} = netstates.convedmaps1{i}*(1-arch.conv_dropout_fraction);
        else
            netstates.conv1_dropout_mask{i} = rand(size(netstates.convedmaps1{i}))>arch.conv_dropout_fraction;
            netstates.convedmaps1{i} = netstates.convedmaps1{i}.*netstates.conv1_dropout_mask{i};
        end   
    end
    
    % pooling
    if strcmp(arch.poolstyle, 'mean')       % mean pooling
        pooled = convn(netstates.convedmaps1{i}, ones(arch.poolscale1,arch.poolscale1)/arch.poolscale1^2, 'valid');
        netstates.pooledmaps1{i} = pooled(1:arch.poolscale1:end, 1:arch.poolscale1:end, :);
    elseif strcmp(arch.poolstyle, 'max')    % max pooling
        netstates.pooledmaps1{i} = zeros(pooleddim1,pooleddim1,batchsize);
        netstates.posinds1{i} = zeros(pooleddim1,pooleddim1,batchsize);
        rows = size(netstates.convedmaps1{i},1)/arch.poolscale1;
        cols = size(netstates.convedmaps1{i},2)/arch.poolscale1;
        for row=1:rows
            for col=1:cols
                patches = netstates.convedmaps1{i}((row-1)*arch.poolscale1+1:row*arch.poolscale1,(col-1)*arch.poolscale1+1:col*arch.poolscale1, :);
                [C1, I1] = max(reshape(patches,arch.poolscale1^2,batchsize));
                netstates.pooledmaps1{i}(row,col,:) = reshape(C1,1,1,batchsize);
                netstates.posinds1{i}(row, col,:) = reshape(I1,1,1,batchsize);
            end
        end
    elseif strcmp(arch.poolstyle, 'stoc_mean') % stochastic mean pooling
        netstates.pooledmaps1{i} = zeros(pooleddim1,pooleddim1,batchsize);
        netstates.posinds1_I1{i} = zeros(pooleddim1,pooleddim1,batchsize);
        netstates.posinds1_I2{i} = zeros(pooleddim1,pooleddim1,batchsize);
        netstates.posinds1_I3{i} = zeros(pooleddim1,pooleddim1,batchsize);
        netstates.posinds1_I4{i} = zeros(pooleddim1,pooleddim1,batchsize);
        rows = size(netstates.convedmaps1{i},1)/arch.poolscale1;
        cols = size(netstates.convedmaps1{i},2)/arch.poolscale1;
        randvals = rand(rows,cols,batchsize);
        
        C1 = netstates.convedmaps1{i}(1:2:end,1:2:end,:);
        C2 = netstates.convedmaps1{i}(2:2:end,1:2:end,:);
        C3 = netstates.convedmaps1{i}(1:2:end,2:2:end,:);
        C4 = netstates.convedmaps1{i}(2:2:end,2:2:end,:);
        I1 = logical(randvals<=0.25); netstates.posinds1_I1{i} = I1;
        I2 = logical((randvals>0.25).*(randvals<=0.5)); netstates.posinds1_I2{i} = I2;
        I3 = logical((randvals>0.5).*(randvals<=0.75)); netstates.posinds1_I3{i} = I3; 
        I4 = logical(randvals>0.75); netstates.posinds1_I4{i} = I4;
        
        netstates.pooledmaps1{i}(I1) = C1(I1);
        netstates.pooledmaps1{i}(I2) = C2(I2);
        netstates.pooledmaps1{i}(I3) = C3(I3);
        netstates.pooledmaps1{i}(I4) = C4(I4);
    elseif strcmp(arch.poolstyle, 'stoc_max')  % stochastic max pooling
        netstates.pooledmaps1{i} = zeros(pooleddim1,pooleddim1,batchsize);
        netstates.posinds1_I1{i} = zeros(pooleddim1,pooleddim1,batchsize);
        netstates.posinds1_I2{i} = zeros(pooleddim1,pooleddim1,batchsize);
        netstates.posinds1_I3{i} = zeros(pooleddim1,pooleddim1,batchsize);
        netstates.posinds1_I4{i} = zeros(pooleddim1,pooleddim1,batchsize);
        rows = size(netstates.convedmaps1{i},1)/arch.poolscale1;
        cols = size(netstates.convedmaps1{i},2)/arch.poolscale1;
        randvals = rand(rows,cols,batchsize);
        
        C1 = netstates.convedmaps1{i}(1:2:end,1:2:end,:)+1e-6;
        C2 = netstates.convedmaps1{i}(2:2:end,1:2:end,:)+1e-6;
        C3 = netstates.convedmaps1{i}(1:2:end,2:2:end,:)+1e-6;
        C4 = netstates.convedmaps1{i}(2:2:end,2:2:end,:)+1e-6;
        Cs = C1+C2+C3+C4;
        probC1 = C1./Cs;probC2 = C2./Cs;probC3 = C3./Cs;probC4 = C4./Cs;
        I1 = logical(randvals<=probC1); netstates.posinds1_I1{i} = I1;
        I2 = logical((randvals>probC1).*(randvals<=probC1+probC2)); netstates.posinds1_I2{i} = I2;
        I3 = logical((randvals>probC1+probC2).*(randvals<=probC1+probC2+probC3)); netstates.posinds1_I3{i} = I3; 
        I4 = logical(randvals>probC1+probC2+probC3); netstates.posinds1_I4{i} = I4;
        if strcmp(arch.mode, 'test')
            netstates.pooledmaps1{i} = probC1.*C1+probC2.*C2+probC3.*C3+probC4.*C4;
        else
            netstates.pooledmaps1{i}(I1) = C1(I1);
            netstates.pooledmaps1{i}(I2) = C2(I2);
            netstates.pooledmaps1{i}(I3) = C3(I3);
            netstates.pooledmaps1{i}(I4) = C4(I4);
        end
    else
        error('Check pool style, only mean or max plooling style is valid.');
    end
    
    % dropout
    if arch.pool_dropout_fraction>0
        if strcmp(arch.mode, 'test')
            netstates.pooledmaps1{i} = netstates.pooledmaps1{i}*(1-arch.pool_dropout_fraction);
        else
            netstates.pool1_dropout_mask{i} = rand(size(netstates.pooledmaps1{i}))>arch.pool_dropout_fraction;
            netstates.pooledmaps1{i} = netstates.pooledmaps1{i}.*netstates.pool1_dropout_mask{i};
        end   
    end
end

%% layer 2
pooleddim2 = (pooleddim1-arch.filterdim2+1)/arch.poolscale2;
netstates.ispos2 = cell(arch.numfilters2, 1);
netstates.convedmaps2 = cell(arch.numfilters2, 1);
netstates.conv2_dropout_mask = cell(arch.numfilters2, 1);
netstates.pooledmaps2 = cell(arch.numfilters2, 1);
netstates.posinds2 = cell(arch.numfilters2, 1);
netstates.featvec = zeros(pooleddim2^2, batchsize);
for i = 1:arch.numfilters2
    % convolution
    inputs2conv2 = zeros(pooleddim1-arch.filterdim2+1, pooleddim1-arch.filterdim2+1, batchsize);
    for j=1:arch.numfilters1
        inputs2conv2 = inputs2conv2 + convn(netstates.pooledmaps1{j}, rot90(modelparas.filters2{j}(:,:,i),2), 'valid');
    end
    inputs2conv2 = inputs2conv2+ modelparas.filters2_bias(i);
    if strcmp(arch.acttype, 'sigm') % sigmoid unit
        netstates.convedmaps2{i} = 1./(1+exp(-inputs2conv2));
    elseif strcmp(arch.acttype, 'relu') % rectified linear unit
        netstates.ispos2{i} = double(inputs2conv2>0);
        netstates.convedmaps2{i} = inputs2conv2.*netstates.ispos2{i};
    end
    
    if arch.conv_dropout_fraction>0
        if strcmp(arch.mode, 'test')
            netstates.convedmaps2{i} = netstates.convedmaps2{i}*(1-arch.conv_dropout_fraction);
        else
            netstates.conv2_dropout_mask{i} = rand(size(netstates.convedmaps2{i}))>arch.conv_dropout_fraction;
            netstates.convedmaps2{i} = netstates.convedmaps2{i}.*netstates.conv2_dropout_mask{i};
        end   
    end
    
    % pooling
    if strcmp(arch.poolstyle, 'mean')       % mean pooling
        pooled = convn(netstates.convedmaps2{i}, ones(arch.poolscale2,arch.poolscale2)/arch.poolscale2^2, 'valid');
        netstates.pooledmaps2{i} = pooled(1:arch.poolscale2:end, 1:arch.poolscale2:end, :);
    elseif strcmp(arch.poolstyle, 'max')    % max pooling
        netstates.pooledmaps2{i} = zeros(pooleddim2,pooleddim2,batchsize);
        netstates.posinds2{i} = zeros(pooleddim2,pooleddim2,batchsize);
        rows = size(netstates.convedmaps2{i},1)/arch.poolscale2;
        cols = size(netstates.convedmaps2{i},2)/arch.poolscale2;
        for row=1:rows
            for col=1:cols
                patches = netstates.convedmaps2{i}((row-1)*arch.poolscale2+1:row*arch.poolscale2,(col-1)*arch.poolscale2+1:col*arch.poolscale2, :);
                [C1, I1] = max(reshape(patches,arch.poolscale2^2,batchsize));
                netstates.pooledmaps2{i}(row,col,:) = reshape(C1,1,1,batchsize);
                netstates.posinds2{i}(row,col,:) = reshape(I1,1,1,batchsize);
            end
        end
    elseif strcmp(arch.poolstyle, 'stoc_mean') % stochastic mean pooling
        netstates.pooledmaps2{i} = zeros(pooleddim2,pooleddim2,batchsize);
        netstates.posinds2_I1{i} = zeros(pooleddim2,pooleddim2,batchsize);
        netstates.posinds2_I2{i} = zeros(pooleddim2,pooleddim2,batchsize);
        netstates.posinds2_I3{i} = zeros(pooleddim2,pooleddim2,batchsize);
        netstates.posinds2_I4{i} = zeros(pooleddim2,pooleddim2,batchsize);
        rows = size(netstates.convedmaps2{i},1)/arch.poolscale2;
        cols = size(netstates.convedmaps2{i},2)/arch.poolscale2;
        randvals = rand(rows,cols,batchsize);
        
        I1 = logical(randvals<=0.25); netstates.posinds2_I1{i} = I1;
        I2 = logical((randvals>0.25).*(randvals<=0.5)); netstates.posinds2_I2{i} = I2;
        I3 = logical((randvals>0.5).*(randvals<=0.75)); netstates.posinds2_I3{i} = I3; 
        I4 = logical(randvals>0.75); netstates.posinds2_I4{i} = I4;
        C1 = netstates.convedmaps2{i}(1:2:end,1:2:end,:);
        C2 = netstates.convedmaps2{i}(2:2:end,1:2:end,:);
        C3 = netstates.convedmaps2{i}(1:2:end,2:2:end,:);
        C4 = netstates.convedmaps2{i}(2:2:end,2:2:end,:);
        
        netstates.pooledmaps2{i}(I1) = C1(I1);
        netstates.pooledmaps2{i}(I2) = C2(I2);
        netstates.pooledmaps2{i}(I3) = C3(I3);
        netstates.pooledmaps2{i}(I4) = C4(I4);
    elseif strcmp(arch.poolstyle, 'stoc_max')  % stochastic max pooling
        netstates.pooledmaps2{i} = zeros(pooleddim2,pooleddim2,batchsize);
        netstates.posinds2_I1{i} = zeros(pooleddim2,pooleddim2,batchsize);
        netstates.posinds2_I2{i} = zeros(pooleddim2,pooleddim2,batchsize);
        netstates.posinds2_I3{i} = zeros(pooleddim2,pooleddim2,batchsize);
        netstates.posinds2_I4{i} = zeros(pooleddim2,pooleddim2,batchsize);
        rows = size(netstates.convedmaps2{i},1)/arch.poolscale2;
        cols = size(netstates.convedmaps2{i},2)/arch.poolscale2;
        randvals = rand(rows,cols,batchsize);
        
        C1 = netstates.convedmaps2{i}(1:2:end,1:2:end,:)+1e-6;
        C2 = netstates.convedmaps2{i}(2:2:end,1:2:end,:)+1e-6;
        C3 = netstates.convedmaps2{i}(1:2:end,2:2:end,:)+1e-6;
        C4 = netstates.convedmaps2{i}(2:2:end,2:2:end,:)+1e-6;
        Cs = C1+C2+C3+C4;
        probC1 = C1./Cs;probC2 = C2./Cs;probC3 = C3./Cs;probC4 = C4./Cs;
        I1 = logical(randvals<=probC1); netstates.posinds2_I1{i} = I1;
        I2 = logical((randvals>probC1).*(randvals<=probC1+probC2)); netstates.posinds2_I2{i} = I2;
        I3 = logical((randvals>probC1+probC2).*(randvals<=probC1+probC2+probC3)); netstates.posinds2_I3{i} = I3; 
        I4 = logical(randvals>probC1+probC2+probC3); netstates.posinds2_I4{i} = I4;
        
        if strcmp(arch.mode, 'test')
            netstates.pooledmaps2{i} = probC1.*C1+probC2.*C2+probC3.*C3+probC4.*C4;
        else
            netstates.pooledmaps2{i}(I1) = C1(I1);
            netstates.pooledmaps2{i}(I2) = C2(I2);
            netstates.pooledmaps2{i}(I3) = C3(I3);
            netstates.pooledmaps2{i}(I4) = C4(I4);
        end
    else
        error('Check pool style, only mean or max plooling style is valid.');
    end
    
    % reshape
     netstates.featvec((i-1)*pooleddim2^2+1:i*pooleddim2^2, :) = reshape(netstates.pooledmaps2{i},pooleddim2^2,batchsize);
end

%% featvec layer to hidden layer
if arch.featvec_dropout_fraction>0
    if strcmp(arch.mode, 'test')
        netstates.featvec = netstates.featvec*(1-arch.featvec_dropout_fraction);
    else
        netstates.featvec_dropout_mask = rand(size(netstates.featvec))>arch.featvec_dropout_fraction;
        netstates.featvec = netstates.featvec.*netstates.featvec_dropout_mask;
    end   
end
inputs2hid = modelparas.featvec2hid_weights*netstates.featvec + repmat(modelparas.hid_bias, 1, batchsize);
netstates.hidstates = 1./(1+exp(-inputs2hid));
% if strcmp(arch.acttype, 'sigm') % sigmoid unit
%     netstates.hidstates = 1./(1+exp(-inputs2hid));
% elseif strcmp(arch.acttype, 'relu') % rectified linear unit
%     netstates.ispos_hid = double(inputs2hid>0);
%     netstates.hidstates = inputs2hid.*netstates.ispos_hid;
% end
if arch.hid_dropout_fraction>0
    if strcmp(arch.mode, 'test')
        netstates.hidstates = netstates.hidstates*arch.hid_dropout_fraction;
    else
        netstates.hid_dropout_mask = rand(size(netstates.hidstates))>arch.hid_dropout_fraction;
        netstates.hidstates = netstates.hidstates.*netstates.hid_dropout_mask;
    end
end

%% hidden layer to output layer
inputs2out = modelparas.hid2out_weights*netstates.hidstates + repmat(modelparas.out_bias,1,batchsize);
inputs2out = inputs2out - repmat(max(inputs2out), arch.outputdim, 1);
netstates.outstates = exp(inputs2out)./repmat(sum(exp(inputs2out)), arch.outputdim, 1);

%% cross entropy loss
loss = -sum(sum(batch_y.*log(netstates.outstates)))/batchsize;

end
