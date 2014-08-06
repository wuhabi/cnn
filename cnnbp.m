function [ grads ] = cnnbp( arch, modelparas, netstates, batch_x, batch_y)
%CNNBP 
%   

batchsize = size(batch_x, 3);
conveddim1 = arch.inputdim-arch.filterdim1+1;
pooleddim1 = conveddim1/arch.poolscale1;
conveddim2 = pooleddim1-arch.filterdim2+1;
pooleddim2 = conveddim2/arch.poolscale2;

%% calculate deltas
inputs2out_delta = netstates.outstates - batch_y; % (outputdim X batchsize)

% derivatives of CE loss w.r.t. inputs2hid
inputs2hid_delta = (modelparas.hid2out_weights'*inputs2out_delta).*netstates.hidstates.*(1-netstates.hidstates); % (hiddim X batchsize)
% if strcmp(arch.acttype, 'sigm') % sigmoid unit
%     inputs2hid_delta = (modelparas.hid2out_weights'*inputs2out_delta).*netstates.hidstates.*(1-netstates.hidstates); % (hiddim X batchsize)
% elseif strcmp(arch.acttype, 'relu') % rectified linear unit
%     inputs2hid_delta = (modelparas.hid2out_weights'*inputs2out_delta).*netstates.ispos_hid;
% end
if arch.hid_dropout_fraction>0
    inputs2hid_delta = inputs2hid_delta.*netstates.hid_dropout_mask;
end

% derivatives of CE loss w.r.t. featvec
featvec_delta = modelparas.featvec2hid_weights' * inputs2hid_delta; % (featvecdim X batchsize)
if arch.featvec_dropout_fraction>0
    featvec_delta = featvec_delta.*netstates.featvec_dropout_mask;
end

% derivatives of CE loss w.r.t. 
pooledmaps2_delta = cell(arch.numfilters2, 1);
convedmaps2_delta = cell(arch.numfilters2, 1);
inputs2conv2_delta = cell(arch.numfilters2, 1);
for i = 1:arch.numfilters2
    % reshape
    pooledmaps2_delta{i} = reshape(featvec_delta((i-1)*pooleddim2^2+1:i*pooleddim2^2, :), pooleddim2, pooleddim2, batchsize);
    
    % upsample
    convedmaps2_delta{i} = zeros(conveddim2, conveddim2, batchsize);
    if strcmp(arch.poolstyle, 'mean')
        for j = 1:batchsize
            convedmaps2_delta{i}(:,:,j) = kron(pooledmaps2_delta{i}(:,:,j), ones(arch.poolscale2)/arch.poolscale2^2);
        end
    elseif strcmp(arch.poolstyle, 'max')
        pooledmaps2_delta_patches = pooledmaps2_delta{i};
        for row = 1:size(pooledmaps2_delta_patches, 1)
            for col = 1:size(pooledmaps2_delta_patches, 2)
                upsample_deltas = pooledmaps2_delta_patches(row,col,:); % (1 X 1 X batchsize)
                upsample_delta_patches = zeros(arch.poolscale2, arch.poolscale2, batchsize);
                inds = netstates.posinds2{i}(row,col,:); % (1 X 1 X batchsize)
                [Is, Js] = ind2sub([arch.poolscale2 arch.poolscale2], inds);
                for j = 1:batchsize
                    upsample_delta_patches(Is(j),Js(j),j) = upsample_deltas(j);
                end
                convedmaps2_delta{i}((row-1)*arch.poolscale2+1:row*arch.poolscale2,(col-1)*arch.poolscale2+1:col*arch.poolscale2,:) = upsample_delta_patches;
            end
        end
    elseif strcmp(arch.poolstyle, 'stoc_mean') || strcmp(arch.poolstyle, 'stoc_max')
        pooledmaps2_delta_patches = pooledmaps2_delta{i};
        temp = zeros(size(pooledmaps2_delta_patches));
        
        temp(netstates.posinds2_I1{i}) = pooledmaps2_delta_patches(netstates.posinds2_I1{i});
        convedmaps2_delta{i}(1:2:end,1:2:end,:) = temp;
        temp = 0*temp;
        temp(netstates.posinds2_I2{i}) = pooledmaps2_delta_patches(netstates.posinds2_I2{i});
        convedmaps2_delta{i}(2:2:end,1:2:end,:) = temp;
        temp = 0*temp;
        temp(netstates.posinds2_I3{i}) = pooledmaps2_delta_patches(netstates.posinds2_I3{i});
        convedmaps2_delta{i}(1:2:end,2:2:end,:) = temp;
        temp = 0*temp;
        temp(netstates.posinds2_I4{i}) = pooledmaps2_delta_patches(netstates.posinds2_I4{i});
        convedmaps2_delta{i}(2:2:end,2:2:end,:) = temp;
    end
    
    if arch.conv_dropout_fraction>0
        convedmaps2_delta{i} = convedmaps2_delta{i}.*netstates.conv2_dropout_mask{i};
    end
    if strcmp(arch.acttype, 'sigm')
        inputs2conv2_delta{i} = convedmaps2_delta{i}.*netstates.convedmaps2{i}.*(1-netstates.convedmaps2{i});
    elseif strcmp(arch.acttype, 'relu')
        inputs2conv2_delta{i} = convedmaps2_delta{i}.*netstates.ispos2{i};
    end
end

pooledmaps1_delta = cell(arch.numfilters1, 1);
convedmaps1_delta = cell(arch.numfilters1, 1);
inputs2conv1_delta = cell(arch.numfilters1, 1);
for i = 1:arch.numfilters1
    % derivatives of CE loss w.r.t. pooledmaps1
    pooledmaps1_delta{i} = zeros(pooleddim1, pooleddim1, batchsize);
    for j = 1:arch.numfilters2
%         pooledmaps1_delta{i} = pooledmaps1_delta{i} + convn(inputs2conv2_delta{j}, rot90(modelparas.filters2{i}(:,:,j),2), 'full');
        pooledmaps1_delta{i} = pooledmaps1_delta{i} + convn(inputs2conv2_delta{j}, modelparas.filters2{i}(:,:,j), 'full');
    end
    
    if arch.pool_dropout_fraction>0
        pooledmaps1_delta{i} = pooledmaps1_delta{i}.*netstates.pool1_dropout_mask{i};
    end
    
    % upsample
    convedmaps1_delta{i} = zeros(conveddim1, conveddim1, batchsize);
    if strcmp(arch.poolstyle, 'mean')
        for j = 1:batchsize
            convedmaps1_delta{i}(:,:,j) = kron(pooledmaps1_delta{i}(:,:,j), ones(arch.poolscale1)/arch.poolscale1^2);
        end
    elseif strcmp(arch.poolstyle, 'max')
        pooledmaps1_delta_patches = pooledmaps1_delta{i};
        for row = 1:size(pooledmaps1_delta_patches, 1)
            for col = 1:size(pooledmaps1_delta_patches, 2)
                upsample_deltas = pooledmaps1_delta_patches(row,col,:);
                upsample_delta_patches = zeros(arch.poolscale1, arch.poolscale1, batchsize);
                inds = netstates.posinds1{i}(row,col,:);
                [Is, Js] = ind2sub([arch.poolscale1 arch.poolscale1], inds);
                for j = 1:batchsize
                    upsample_delta_patches(Is(j),Js(j),j) = upsample_deltas(j);
                end
                convedmaps1_delta{i}((row-1)*arch.poolscale1+1:row*arch.poolscale1,(col-1)*arch.poolscale1+1:col*arch.poolscale1, :) = upsample_delta_patches;
            end
        end
    elseif strcmp(arch.poolstyle, 'stoc_mean') || strcmp(arch.poolstyle, 'stoc_max')
        pooledmaps1_delta_patches = pooledmaps1_delta{i};
        temp = zeros(size(pooledmaps1_delta_patches));
        
        temp(netstates.posinds1_I1{i}) = pooledmaps1_delta_patches(netstates.posinds1_I1{i});
        convedmaps1_delta{i}(1:2:end,1:2:end,:) = temp;
        temp = 0*temp;
        temp(netstates.posinds1_I2{i}) = pooledmaps1_delta_patches(netstates.posinds1_I2{i});
        convedmaps1_delta{i}(2:2:end,1:2:end,:) = temp;
        temp = 0*temp;
        temp(netstates.posinds1_I3{i}) = pooledmaps1_delta_patches(netstates.posinds1_I3{i});
        convedmaps1_delta{i}(1:2:end,2:2:end,:) = temp;
        temp = 0*temp;
        temp(netstates.posinds1_I4{i}) = pooledmaps1_delta_patches(netstates.posinds1_I4{i});
        convedmaps1_delta{i}(2:2:end,2:2:end,:) = temp;
    end
    
    if arch.conv_dropout_fraction>0
        convedmaps1_delta{i} = convedmaps1_delta{i}.*netstates.conv1_dropout_mask{i};
    end
    if strcmp(arch.acttype, 'sigm')
        inputs2conv1_delta{i} = convedmaps1_delta{i}.*netstates.convedmaps1{i}.*(1-netstates.convedmaps1{i});
    elseif strcmp(arch.acttype, 'relu')
        inputs2conv1_delta{i} = convedmaps1_delta{i}.*netstates.ispos1{i};
    end
end

%% calculate gradients

% deriratives of CE loss w.r.t. out_bias
grads.out_bias_grad = sum(inputs2out_delta, 2)/batchsize;
% disp('back-propagation gradients of out_bias:');
% disp(grads.out_bias_grad);

% deriratives of CE loss w.r.t. hid2out_weights
grads.hid2out_weights_grad = inputs2out_delta*netstates.hidstates'/batchsize;
% disp('back-propagation gradients of hid2out_weights:');
% disp(grads.hid2out_weights_grad);

% deriratives of CE loss w.r.t. hid_bias
grads.hid_bias_grad = sum(inputs2hid_delta, 2)/batchsize;
% disp('back-propagation gradients of hid_bias:');
% disp(grads.hid_bias_grad);

% deriratives of CE loss w.r.t. featvec2hid_weights
grads.featvec2hid_weights_grad = inputs2hid_delta*netstates.featvec'/batchsize;
% disp('back-propagation gradients of featvec2hid_weights:');
% disp(grads.featvec2hid_weights_grad);

% deriratives of CE loss w.r.t. filters2
grads.filters2_grad = cell(arch.numfilters1, 1);
for i = 1:arch.numfilters1
    grads.filters2_grad{i} = zeros(arch.filterdim2, arch.filterdim2, arch.numfilters2);
    for j = 1:arch.numfilters2
        grads.filters2_grad{i}(:,:,j) = convn(netstates.pooledmaps1{i}, flipalldim(inputs2conv2_delta{j}), 'valid')/batchsize;
    end
end
% disp('back-propagation gradients of filters2:');
% disp(grads.filters2_grad);

% deriratives of CE loss w.r.t. filters2_bias
grads.filters2_bias_grad = zeros(arch.numfilters2, 1);
for i = 1:arch.numfilters2
    grads.filters2_bias_grad(i) = sum(inputs2conv2_delta{i}(:))/batchsize;
end
% disp('back-propagation gradients of filters2_bias:');
% disp(grads.filters2_bias_grad);

% deriratives of CE loss w.r.t. filters1
grads.filters1_grad = zeros(arch.filterdim1, arch.filterdim1, arch.numfilters1);
% deriratives of CE loss w.r.t. filters1_bias
grads.filters1_bias_grad = zeros(arch.numfilters1, 1);
for i = 1:arch.numfilters1
    grads.filters1_grad(:,:,i) = convn(batch_x, flipalldim(inputs2conv1_delta{i}), 'valid')/batchsize;
    grads.filters1_bias_grad(i) = sum(inputs2conv1_delta{i}(:))/batchsize;
end
% disp('back-propagation gradients of filters1_bias:');
% disp(grads.filters1_bias_grad);
% disp('back-propagation gradients of filters1:');
% disp(grads.filters1_grad);

end

