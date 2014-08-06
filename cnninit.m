function paras = cnninit( arch )
%CNNINIT 
%   

paras.filters1 = 0.1*randn(arch.filterdim1, arch.filterdim1, arch.numfilters1);
paras.filters1_bias = zeros(arch.numfilters1, 1);

convdim1 = arch.inputdim - arch.filterdim1 +1;
assert(mod(convdim1,arch.poolscale1)==0, 'poolscale1 must divide convdim1');
pooleddim1 = convdim1/arch.poolscale1;

paras.filters2 = cell(arch.numfilters1, 1);
for i = 1:arch.numfilters1
    paras.filters2{i} = 0.1*randn(arch.filterdim2, arch.filterdim2, arch.numfilters2);
end
paras.filters2_bias = zeros(arch.numfilters2, 1);

convdim2 = pooleddim1 - arch.filterdim2 +1;
assert(mod(convdim2,arch.poolscale2)==0, 'poolscale2 must divide convdim2');
pooleddim2 = convdim2/arch.poolscale2;

paras.featvec2hid_weights = 0.1*randn(arch.hiddim, arch.numfilters2*pooleddim2^2);
paras.hid_bias = zeros(arch.hiddim, 1);
paras.hid2out_weights = 0.1*randn(arch.outputdim, arch.hiddim);
paras.out_bias = zeros(arch.outputdim, 1);

end

