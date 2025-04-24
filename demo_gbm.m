%% Initialization
clear
clc
warning off;

datasets = {'GBM1.mat'};
results = struct();
save_folder = 'E:\run_figgbm_testgit\';

if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

%% Main Loop: Datasets and Runs
for run_idx = 1:10  
    for dataset_idx = 1
        dataset_name = datasets{dataset_idx};
        dataset_fieldname = erase(dataset_name, '.mat');  
        disp(['Processing dataset: ', dataset_name, ' (Run ', num2str(run_idx), ')']);

        load(dataset_name);

        %% Data Preprocessing
        Data{1,1} = ProgressData(Gene);
        Data{1,2} = ProgressData(Methy);
        Data{1,3} = ProgressData(Mirna);
        for v = 1:3
            Data{v} = normalize_data(Data{v});
        end
        label = table2array(Response(:,2:end));

        %% Clustering Evaluation
        best_p_values_all = [];
        best_params_all = {};
        all_sorted_p_values = [];

        for numclass = 5
            numsamples = size(Data{1,1},1);
            Y = table2array(Response(:,2:end));

            %% Compute kernel matrices
            H_num = 5;
            [KH, HP, num_kernel] = pre_s(Data, numclass, H_num,label);

            %% Parameter Grid
            r1 = 0.1:0.2:1;
            r2 = (0.1:0.1:0.9) * num_kernel;
            r3 = 0.1:0.2:1;
            r4 = 0.1:0.2:1;
            num_best = 1;  % number of top results to keep
            best_p_values = inf(1, num_best);
            best_r1_values = NaN(1, num_best);
            best_r2_values = NaN(1, num_best);
            best_r3_values = NaN(1, num_best);
            best_r4_values = NaN(1, num_best);
            best_groups = cell(1, num_best);

            %% Grid Search for Best Parameters
            for r1Index = 1:length(r1)
                r1Temp = r1(r1Index);
                for r2Index = 1:length(r2)
                    r2Temp = ceil(r2(r2Index));
                    for r3Index = 1:length(r3)
                        r3Temp = r3(r3Index);
                        for r4Index = 1:length(r4)
                            r4Temp = r4(r4Index);

                            disp(['Testing r1 = ', num2str(r1Temp), ...
                                ', r2 = ', num2str(r2Temp), ...
                                ', r3 = ', num2str(r3Temp), ...
                                ', r4 = ', num2str(r4Temp)]);

                            %% Optimization and Clustering
                            [F, obj] = myfun(KH, HP, H_num, numclass, r1Temp, r2Temp, r3Temp, r4Temp);
                            pre = kmeans(real(F), numclass, 'maxiter', 100, 'replicates', 20, 'emptyaction', 'singleton');
                            group = num2str(pre);
                            group1 = num2cell(group);

                            %% Compute p-value
                            p = MatSurv(label(:,1), label(:,2), group1, 'CensorLineLength', 0, 'LineWidth', 1.2, 'CensorLineWidth', 1, 'NoPlot', true);
                            disp(['p-value = ', num2str(p)]);

                            %% Update Best Result
                            if p > 0
                                [max_p, max_idx] = max(best_p_values);
                                if p < max_p
                                    best_p_values(max_idx) = p;
                                    best_r1_values(max_idx) = r1Temp;
                                    best_r2_values(max_idx) = r2Temp;
                                    best_r3_values(max_idx) = r3Temp;
                                    best_r4_values(max_idx) = r4Temp;
                                    best_groups{max_idx} = group1;
                                end
                            end
                        end
                    end
                end
            end

            %% Record Best Results for Current Class Number
            best_p_values_all = [best_p_values_all; numclass, min(best_p_values)];
            best_params_all{end+1} = [best_r1_values, best_r2_values, best_r3_values, best_r4_values];

            [sorted_p_values, sorted_indices] = sort(best_p_values, 'ascend');
            all_sorted_p_values{end+1} = sorted_p_values;

            disp(['Best p-value for numclass = ', num2str(numclass), ': ', num2str(min(best_p_values))]);

            %% Survival Plotting and Saving
            for i = 1:num_best
                p = MatSurv(label(:,1), label(:,2), best_groups{i}, ...
                    'CensorLineLength', 0, 'LineWidth', 1.5, 'CensorLineWidth', 15);
                disp(['Sorted p-value for run ', num2str(i), ': ', num2str(p)]);

                fig = gcf;
                set(fig, 'PaperPositionMode', 'auto');
                filename_base = fullfile(save_folder, sprintf('%s_numclass%d_run%d_best%d', ...
                    dataset_fieldname, numclass, run_idx, i));

                savefig(fig, [filename_base, '.fig']);
                print(fig, [filename_base, '.eps'], '-depsc', '-r600');
                close(fig);
            end
        end

        %% Store Results
        results.(dataset_fieldname)(run_idx).best_p_values_all = best_p_values_all;
        results.(dataset_fieldname)(run_idx).best_params_all = best_params_all;
        results.(dataset_fieldname)(run_idx).all_sorted_p_values = all_sorted_p_values;

        disp(['Results for ', dataset_name, ' (Run ', num2str(run_idx), '):']);
        disp(array2table(best_p_values_all, 'VariableNames', {'numclass', 'best_p_value'}));
    end
end

%% Final Summary Across All Datasets
disp('Final Results Summary:');
for dataset_idx = 1:length(datasets)
    dataset_fieldname = erase(datasets{dataset_idx}, '.mat');
    disp(['Dataset: ', dataset_fieldname]);
    for run_idx = 1:10
        disp(['Run ', num2str(run_idx)]);
        disp(array2table(results.(dataset_fieldname)(run_idx).best_p_values_all, 'VariableNames', {'numclass', 'best_p_value'}));
    end
end
