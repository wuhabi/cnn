function [ numgrads ] = computenumgrad(arch, modelparas, batch_x, batch_y)
%NUMGRADCHECK 
%   

epsilon = 1e-4;

numgrads.out_bias_grad = zeros(arch.outputdim, 1);
for i = 1:length(modelparas.out_bias)
    theta_pos = modelparas; theta_neg = modelparas;
    theta_pos.out_bias(i) = theta_pos.out_bias(i) + epsilon;
    theta_neg.out_bias(i) = theta_neg.out_bias(i) - epsilon;
    
    [~, loss_pos] = cnnff(theta_pos, arch, batch_x, batch_y);
    [~, loss_neg] = cnnff(theta_neg, arch, batch_x, batch_y);
    numgrads.out_bias_grad(i) = (loss_pos-loss_neg)/(2*epsilon);
end
disp('numerical gradients of out_bias:');
disp(numgrads.out_bias_grad);

% numgrads.hid2out_weights_grad = zeros(size(modelparas.hid2out_weights));
% for i = 1:size(modelparas.hid2out_weights, 1)
%     for j = 1:size(modelparas.hid2out_weights, 2)
%         theta_pos = modelparas; theta_neg = modelparas;
%         theta_pos.hid2out_weights(i,j) = theta_pos.hid2out_weights(i,j) + epsilon;
%         theta_neg.hid2out_weights(i,j) = theta_neg.hid2out_weights(i,j) - epsilon;
%         [~, loss_pos] = cnnff(theta_pos, arch, batch_x, batch_y);
%         [~, loss_neg] = cnnff(theta_neg, arch, batch_x, batch_y);
%         numgrads.hid2out_weights_grad(i,j) = (loss_pos-loss_neg)/(2*epsilon);
%     end
% end
% disp('numerical gradients of hid2out_weights:');
% disp(numgrads.hid2out_weights_grad);

% numgrads.hid_bias_grad = zeros(arch.hiddim, 1);
% for i = 1:length(modelparas.hid_bias)
%     theta_pos = modelparas; theta_neg = modelparas;
%     theta_pos.hid_bias(i) = theta_pos.hid_bias(i) + epsilon;
%     theta_neg.hid_bias(i) = theta_neg.hid_bias(i) - epsilon;
%     
%     [~, loss_pos] = cnnff(theta_pos, arch, batch_x, batch_y);
%     [~, loss_neg] = cnnff(theta_neg, arch, batch_x, batch_y);
%     numgrads.hid_bias_grad(i) = (loss_pos-loss_neg)/(2*epsilon);
% end
% disp('numerical gradients of hid_bias:');
% disp(numgrads.hid_bias_grad);
% 
% numgrads.featvec2hid_weights_grad = zeros(size(modelparas.featvec2hid_weights));
% for i = 1:size(modelparas.featvec2hid_weights, 1)
%     for j = 1:size(modelparas.featvec2hid_weights, 2)
%         theta_pos = modelparas; theta_neg = modelparas;
%         theta_pos.featvec2hid_weights(i,j) = theta_pos.featvec2hid_weights(i,j) + epsilon;
%         theta_neg.featvec2hid_weights(i,j) = theta_neg.featvec2hid_weights(i,j) - epsilon;
%         [~, loss_pos] = cnnff(theta_pos, arch, batch_x, batch_y);
%         [~, loss_neg] = cnnff(theta_neg, arch, batch_x, batch_y);
%         numgrads.featvec2hid_weights_grad(i,j) = (loss_pos-loss_neg)/(2*epsilon);
%     end
% end
% disp('numerical gradients of featvec2hid_weights:');
% disp(numgrads.featvec2hid_weights_grad);

% numgrads.filters2_bias_grad = zeros(size(modelparas.filters2_bias));
% for i = 1:length(modelparas.filters2_bias)
%     theta_pos = modelparas; theta_neg = modelparas;
%     theta_pos.filters2_bias(i) = theta_pos.filters2_bias(i) + epsilon;
%     theta_neg.filters2_bias(i) = theta_neg.filters2_bias(i) - epsilon;
%     [~, loss_pos] = cnnff(theta_pos, arch, batch_x, batch_y);
%     [~, loss_neg] = cnnff(theta_neg, arch, batch_x, batch_y);
%     numgrads.filters2_bias_grad(i) = (loss_pos-loss_neg)/(2*epsilon);
% end
% disp('numerical gradients of filters2_bias:');
% disp(numgrads.filters2_bias_grad);
% 
% numgrads.filters2_grad = cell(arch.numfilters1, 1);
% for i = 1:arch.numfilters1
%     numgrads.filters2_grad{i} = zeros(arch.filterdim2, arch.filterdim2, arch.numfilters2);
%     for j = 1:arch.numfilters2
%         for row = 1:arch.filterdim2
%             for col = 1:arch.filterdim2
%                 theta_pos = modelparas; theta_neg = modelparas;
%                 theta_pos.filters2{i}(row,col,j) = theta_pos.filters2{i}(row,col,j) + epsilon;
%                 theta_neg.filters2{i}(row,col,j) = theta_neg.filters2{i}(row,col,j) - epsilon;
%                 [~, loss_pos] = cnnff(theta_pos, arch, batch_x, batch_y);
%                 [~, loss_neg] = cnnff(theta_neg, arch, batch_x, batch_y);
%                 numgrads.filters2_grad{i}(row,col,j) = (loss_pos-loss_neg)/(2*epsilon);
%             end
%         end
%     end
% end
% disp('numerical gradients of filters2:');
% disp(numgrads.filters2_grad);
% 
% numgrads.filters1_bias_grad = zeros(arch.numfilters1, 1);
% for i = 1:arch.numfilters1
%     theta_pos = modelparas; theta_neg = modelparas;
%     theta_pos.filters1_bias(i) = theta_pos.filters1_bias(i) + epsilon;
%     theta_neg.filters1_bias(i) = theta_neg.filters1_bias(i) - epsilon;
%     [~, loss_pos] = cnnff(theta_pos, arch, batch_x, batch_y);
%     [~, loss_neg] = cnnff(theta_neg, arch, batch_x, batch_y);
%     numgrads.filters1_bias_grad(i) = (loss_pos-loss_neg)/(2*epsilon);
% end
% disp('numerical gradients of filters1_bias:');
% disp(numgrads.filters1_bias_grad);
% 
% numgrads.filters1_grad = zeros(arch.filterdim1, arch.filterdim1, arch.numfilters1);
% for i = 1:arch.numfilters1
%     for row = 1:arch.filterdim1
%         for col = 1:arch.filterdim1
%             theta_pos = modelparas; theta_neg = modelparas;
%             theta_pos.filters1(row,col,i) = theta_pos.filters1(row,col,i) + epsilon;
%             theta_neg.filters1(row,col,i) = theta_neg.filters1(row,col,i) - epsilon;
%             [~, loss_pos] = cnnff(theta_pos, arch, batch_x, batch_y);
%             [~, loss_neg] = cnnff(theta_neg, arch, batch_x, batch_y);
%             numgrads.filters1_grad(row,col,i) = (loss_pos-loss_neg)/(2*epsilon);
%         end
%     end
% end
% disp('numerical gradients of filters1:');
% disp(numgrads.filters1_grad);

end

