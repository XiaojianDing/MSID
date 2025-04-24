function [KH, HP, num_kernel] = pre_s(fea, num_cluster, H_num, label)
num_view = length(fea);
num_sample = size(fea{1}, 1);

% normalize data
data_normalized = cell(num_view, 1);
for v = 1:num_view
    data_normalized{v} = normalize_data(fea{v});
end

all_kernels = cell(num_view, 60);
all_p_values = cell(num_view, 60);

for v = 1:num_view
    for kernel_idx = 1:60
        K_gussian = kernel_matrix(data_normalized{v}, 'RBF4_kernel', 2^(kernel_idx));
        cluster_idx = litekmeans(K_gussian, num_cluster, 'MaxIter', 100, 'Replicates', 10);
        group = num2str(cluster_idx);
        group1 = num2cell(group);
        [p_value] = MatSurv(label(:,1), label(:,2), group1, ...
            'CensorLineLength', 0, 'NoPlot', true, 'print', false);
        all_kernels{v, kernel_idx} = K_gussian;
        all_p_values{v, kernel_idx} = p_value;
    end
end

top_kernel_indices = zeros(num_view, 5);
for v = 1:num_view
    current_p_values = cell2mat(all_p_values(v, :));
    valid_indices = current_p_values > 0;
    filtered_p_values = current_p_values(valid_indices);
    
    [~, sorted_indices] = sort(filtered_p_values, 'ascend');
    top_kernel_indices(v, :) = sorted_indices(1:5);
end

for v = 1:num_view
    options.KernelType = 'Polynomial';
    options.d = 3;
    KH(:, :, 1 + (v - 1) * 9) = construct_kernel(data_normalized{v}, [], options);

    options.KernelType = 'Linear';
    KH(:, :, 2 + (v - 1) * 9) = construct_kernel(data_normalized{v}, [], options);

    options.KernelType = 'Sigmoid';
    options.c = 0;
    options.d = 0.1;
    KH(:, :, 3 + (v - 1) * 9) = construct_kernel(data_normalized{v}, [], options);

    options.KernelType = 'InvPloyPlus';
    options.c = 0.01;
    options.d = 1;
    KH(:, :, 4 + (v - 1) * 9) = construct_kernel(data_normalized{v}, [], options);

    options.KernelType = 'Gaussian';
    KH(:, :, 5 + (v - 1) * 9) = construct_kernel(data_normalized{v}, [], options);
    
    for k = 1:4
        KH(:, :, 5 + k + (v - 1) * 9) = kernel_matrix(data_normalized{v}, 'RBF4_kernel', 2^(top_kernel_indices(v, k)));
    end
end

KH_fused = zeros(size(KH, 1), size(KH, 2), num_view);
num_kernels_per_view=9;
for v = 1:num_view
    K_view = KH(:, :, (v-1)*num_kernels_per_view + 1 : v*num_kernels_per_view);
    T = data_normalized{v} * data_normalized{v}';
    eta = ones(num_kernels_per_view, 1) / num_kernels_per_view;
    alpha = 0.01;
    max_iter = 100; 
    for iter = 1:max_iter
        K_eta = zeros(size(K_view, 1), size(K_view, 2));
        for m = 1:num_kernels_per_view
            K_eta = K_eta + eta(m) * K_view(:, :, m);
        end
        numerator = trace(K_eta * T');
        denominator = sqrt(trace(K_eta * K_eta') * trace(T * T'));
        A = numerator / denominator;

        grad = zeros(num_kernels_per_view, 1);
        for m = 1:num_kernels_per_view
            grad(m) = (trace(K_view(:, :, m) * T') * denominator - numerator * trace(K_view(:, :, m) * K_eta')) / (denominator^3);
        end
        eta = eta + alpha * grad;
        eta = eta / norm(eta);
    end
    K_fused = zeros(size(K_view, 1), size(K_view, 2));
    for m = 1:num_kernels_per_view
        K_fused = K_fused + eta(m) * K_view(:, :, m);
    end
    KH_fused(:, :, v) = K_fused;
end
KH = cat(3, KH, KH_fused);

num_kernels = size(KH, 3);
performance_scores = zeros(num_kernels, 3);

for i = 1:num_kernels
    indx = litekmeans(KH(:,:,i), num_cluster, 'MaxIter', 100, 'Replicates', 10);
    group = num2str(indx);
    group = num2cell(group);
    [p] = MatSurv(label(:,1), label(:,2), group, 'CensorLineLength', 0, 'NoPlot', true);
    P(i) = p;  
    D = eigs(KH(:,:,i), num_cluster);
    r = sqrt(D(1)) / sum(sqrt(D));
    diversity(i) = -log(r) / log(num_cluster);
    performance_scores(i,1) = mean(silhouette(KH(:,:,i), indx));
    performance_scores(i,2) = calinski_harabasz(KH(:,:,i), indx);
    performance_scores(i,3) = 1 / (1 + davies_bouldin(KH(:,:,i), indx)); 
end
sigma2 = var(performance_scores(:));
performance_similarity = exp(-pdist2(performance_scores, performance_scores) .^ 2 / sigma2);
for i = 1:num_kernels
    similarity_scores = performance_similarity(i, :);
end

P(P == 0) = min(P(P > 0)) * 0.1;
accuracy_score = (1./P - 1/max(P)) ./ (1/min(P) - 1/max(P));
diversity = (diversity - min(diversity)) / (max(diversity) - min(diversity));
accuracy_score = (accuracy_score - min(accuracy_score)) / (max(accuracy_score) - min(accuracy_score));
similarity_scores=(similarity_scores-min(similarity_scores))/(max(similarity_scores)-min(similarity_scores));
lambda = 2;
w_diversity = exp(lambda * diversity) ./ (exp(lambda * diversity) + exp(lambda * accuracy_score)+ exp(lambda * similarity_scores));
w_accuracy = exp(lambda * accuracy_score) ./ (exp(lambda * diversity) + exp(lambda * accuracy_score)+ exp(lambda * similarity_scores));
w_similarity=exp(lambda * similarity_scores) ./ (exp(lambda * diversity) + exp(lambda * accuracy_score)+ exp(lambda * similarity_scores));
score = w_diversity .* diversity + w_accuracy .* accuracy_score+w_similarity .* similarity_scores;

[~, sorted_indices] = maxk(score, 20);

KH = KH(:,:,sorted_indices);
num_kernel = length(sorted_indices);

% normalize kernels
KH = knorm(kcenter(KH));

for v = 1:num_kernel
    KH(:,:,v) = (KH(:,:,v) + KH(:,:,v)') / 2;
    for d = 1:H_num
        k = d * num_cluster;
        [Hp{d}, ~] = eigs(KH(:,:,v), k, 'la');    
    end
    HP{v} = Hp;
end
end

function ch_index = calinski_harabasz(K, labels)
    n = size(K, 1);  
    k = length(unique(labels));  
    cluster_centers = zeros(k, size(K, 2));  
    cluster_sizes = zeros(k, 1);  
    overall_mean = mean(K, 1);  

    W = 0;
    B = 0;
    for c = 1:k
        cluster_points = K(labels == c, :);
        cluster_sizes(c) = size(cluster_points, 1);
        if cluster_sizes(c) > 0
            cluster_centers(c, :) = mean(cluster_points, 1);
            W = W + sum(vecnorm(cluster_points - cluster_centers(c, :), 2, 2).^2);
            B = B + cluster_sizes(c) * norm(cluster_centers(c, :) - overall_mean)^2;
        end
    end


    ch_index = (B / (k - 1)) / (W / (n - k));
end

function db_index = davies_bouldin(K, labels)
    k = length(unique(labels)); 
    cluster_centers = zeros(k, size(K, 2));  
    cluster_sizes = zeros(k, 1);  
    s = zeros(k, 1);  


    for c = 1:k
        cluster_points = K(labels == c, :);
        cluster_sizes(c) = size(cluster_points, 1);
        if cluster_sizes(c) > 0
            cluster_centers(c, :) = mean(cluster_points, 1);
            s(c) = mean(vecnorm(cluster_points - cluster_centers(c, :), 2, 2));  % 计算平均距离
        end
    end
    R = zeros(k, k);
    for i = 1:k
        for j = 1:k
            if i ~= j
                d = norm(cluster_centers(i, :) - cluster_centers(j, :));  % 簇间距离
                R(i, j) = (s(i) + s(j)) / d;
            end
        end
    end
    db_index = mean(max(R, [], 2)); 
end


