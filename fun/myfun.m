function  [F, obj] = myfun(KH, HP, H_num, numclass, alpha, K_select,alpha_1,alpha_2)

%% Initialization
[num_sample, ~, num_kernel] = size(KH);
beta = sqrt(ones(1, H_num)/H_num);
mu1 = sqrt(ones(num_kernel, 1)/num_kernel);
mu2 = sqrt(ones(num_kernel, 1)/num_kernel);
mu3 = sqrt(ones(num_kernel, 1)/num_kernel);


for v = 1 : num_kernel
    sum_KH = KH(:, :, v);
    [S{v}, ~] = update_graph(-sum_KH, 15);
end

maxIter = 50;

%%

L=eye(num_sample)-1/num_sample*ones(num_sample);
K_sum=zeros(num_sample,num_sample);
K=cell(num_kernel,1);
gama1 = sqrt(ones(num_kernel,1)/(num_kernel));
for v=1:num_kernel
    K_base=KH(:,:,v);
    K{v} = L * K_base * L;
    K_sum=K_sum+mu3(v)*gama1(v)*K{v};
end
% H
opt.disp = 0;
[T, ~] = eigs(K_sum, numclass, 'la', opt);
%%


flag = 1;
iter = 0;


while flag
    iter = iter + 1;

    %% Update F
    tmp = zeros(num_sample);
    for v = 1 : num_kernel
        Hp = HP{v};
        for d = 1 : H_num
            tmp = tmp + alpha_2 * mu1(v)*beta(d) * Hp{d}*Hp{d}';
        end
        tmp = tmp + alpha * mu2(v) * S{v};
    end
    tmp=tmp+alpha_1*(T*T')/2;
    [F,~] = eigs(tmp, numclass, 'la');

    %% Update gamma
    f_1 = zeros(num_kernel, 1);
    for v = 1 : num_kernel
        tmp = zeros(num_sample);
        Hp = HP{v};
        for d = 1 : H_num
            tmp = tmp + beta(d)*Hp{d}*Hp{d}';
        end
        f_1(v) = trace(alpha*F'*S{v}*F+alpha_2*F'*tmp*F+T' * K{v} * T);
    end
    dis = f_1 ./ norm(f_1);
    [~, gamma] = selec_max(dis, num_kernel, K_select);

    %% update T
    opts = [];  opts.info = 1;
    opts.gtol = 1e-5;
    X=T;
    A=-K_sum;
    G=-alpha_1*F;
    [T, ~] = FOForth(X,G,@fun,opts,A,G);

    %% update K_sum
    K_sum=zeros(num_sample,num_sample);
    for v=1:num_kernel
        K_sum=K_sum+mu3(v)*gamma(v)*K{v};
    end

    %% Update omga
    f_2 = zeros(num_kernel, 1);
    for v = 1 : num_kernel
        tmp = zeros(num_sample);
        Hp = HP{v};
        for d = 1 : H_num
            tmp = tmp + gamma(v)*beta(d)*Hp{d}*Hp{d}';
        end
        f_2(v) = trace(alpha_2*F'*tmp*F);
    end
    mu1 = f_2 ./ norm(f_2);

    %% Update nu
    f_3 = zeros(num_kernel, 1);
    for v = 1 : num_kernel
        f_3(v) = trace(alpha*F'*gamma(v)*S{v}*F);
    end
    mu2 = f_3 ./ norm(f_3);
    %% Update mu
    f_3 = zeros(num_kernel, 1);
    for v = 1 : num_kernel
        f_3(v) = trace(T' * K{v} * T);
    end
    mu3 = f_3 ./ norm(f_3);

    %% Update beta
    f_4 = zeros(1, H_num);
    for d = 1 : H_num
        tmp = zeros(num_sample);
        for v = 1 : num_kernel
            Hs = HP{v}{d};
            tmp = tmp + mu1(v)*Hs*Hs';
        end
        f_4(d) = trace(alpha_2*F'*tmp*F);
    end
    beta = f_4 ./ norm(f_4);


    %% Cal obj
    obj1 = 0;
    obj2 = 0;
    obj3 = 0;
    for v = 1 : num_kernel
        obj1 = obj1 + norm(F*F'-mu2(v)*S{v}, 'fro');
        Hp = HP{v};
        % 计算 Tr(T' * A_v * T)
        for d = 1 : H_num
            obj2 = obj2 + norm(F*F'-mu1(v)*beta(d)*Hp{d}*Hp{d}', 'fro');
        end
    end
    term1 = trace(T' * K_sum * T);
    term2 = trace(F' * T);
    obj3=-alpha_1*term2-term1;
    obj(iter) = alpha*obj1 + alpha_2*obj2+obj3;

    if (iter>2) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-6 || iter>maxIter)
        flag =0;
    end
end
end

function [vec, idx] = selec_max(alpha, num_kernel, k)
vec = zeros(num_kernel, 1);
idx = zeros(num_kernel, 1);
for i = 1 : k
    col = find(alpha==max(alpha));
    vec(col) = alpha(col);
    alpha(col) = 0;
    idx(col) = 1;
end

end
function [funX, F] = fun(X,A,G)
F = A * X + G;
funX = sum(sum(X.* F));
end