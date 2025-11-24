function [KH, HP, num_kernel] = fix_preprocess(fea, numclass, dim_c,label)
num_view = length(fea);
num_sample = size(fea{1}, 1);

% normalize data
fea_normalized = cell(num_view, 1);
for v = 1 : num_view
    fea_normalized{v} = normalize_fea(fea{v});
end

% construct kernels
num_kernel = num_view * 5;
KH = zeros(num_sample, num_sample, num_kernel);
for v = 1 : num_view
    options.KernelType = 'Gaussian';
    KH( : , : , 1 + (v - 1) * 5) = construct_kernel(fea_normalized{v}, [], options);
    options.KernelType = 'Polynomial';
    options.d = 3;
    KH( : , : , 2 + (v - 1) * 5) = construct_kernel(fea_normalized{v}, [], options);
    options.KernelType = 'Linear';
    KH( : , : , 3 + (v - 1) * 5) = construct_kernel(fea_normalized{v}, [], options);
    options.KernelType = 'Sigmoid';
    options.c = 0;
    options.d = 0.1;
    KH( : , : , 4 + (v - 1) * 5) = construct_kernel(fea_normalized{v}, [], options);
    options.KernelType = 'InvPloyPlus';
    options.c = 0.01;
    options.d = 1;
    KH( : , : , 5 + (v - 1) * 5) = construct_kernel(fea_normalized{v}, [], options);
end

% normalize kernels
KH = knorm(kcenter(KH));

num_kernels = size(KH, 3);
diversity = zeros(num_kernels, 1);
P = zeros(num_kernels, 1);  % 存储p-value

for i = 1:num_kernels
    % 计算聚类索引
    indx = litekmeans(KH(:,:,i), numclass, 'MaxIter', 100, 'Replicates', 10);
    group = num2str(indx);
    group = num2cell(group);

    % 计算 p-value
    [p] = MatSurv(label(:,1), label(:,2), group, 'CensorLineLength', 0, 'NoPlot', true);
    P(i) = p;  % 存储 p 值

    % 计算多样性
    D = eigs(KH(:,:,i), numclass);
    r = sqrt(D(1)) / sum(sqrt(D));
    diversity(i) = -log(r) / log(numclass);
end

P(P == 0) = min(P(P > 0)) * 0.1; % 避免 1/P 计算问题
accuracy_score = (1./P - 1/max(P)) ./ (1/min(P) - 1/max(P));

% 归一化
diversity = (diversity - min(diversity)) / (max(diversity) - min(diversity));
accuracy_score = (accuracy_score - min(accuracy_score)) / (max(accuracy_score) - min(accuracy_score));

lambda = 1;  % 影响权重分布的平滑度
w_diversity = exp(lambda * diversity) ./ (exp(lambda * diversity) + exp(lambda * accuracy_score));
w_accuracy = exp(lambda * accuracy_score) ./ (exp(lambda * diversity) + exp(lambda * accuracy_score));



score = w_diversity .* diversity + w_accuracy .* accuracy_score;

[~, selected_indices] = maxk(score, 10);
KH = KH(:,:,selected_indices);
num_kernel = length(selected_indices);

for v = 1 : num_kernel
    KH(:,:,v) = (KH(:,:,v)+KH(:,:,v)')/2;
    for d = 1 : dim_c
        k = d * numclass;
        [Hp{d}, ~] = eigs(KH(:,:,v), k, 'la');    
    end
    HP{v} = Hp;
end




end
