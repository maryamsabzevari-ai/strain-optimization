function [Error_vec, res_vec ] = std_stat()
pr_reas = [3,54,56,75,82];
pr_vals = [0:0.1:1 2:10];
pr_num = 5;
tol_vec = 500:500:10000;
[r,Err_space] = size(tol_vec);
iterations = 50;
s = RandStream('mlfg6331_64');
x2 = zeros(iterations,pr_num);
res_vec = zeros(iterations,Err_space);
Error_vec = zeros(iterations,Err_space);
conc_res = {};
for i = 1:iterations
    x2(i,:) = datasample(s,pr_vals,5,'Replace',true);
    for tol_cnt = 1:Err_space
        [concentration_perturb,Error] = Main_Module(pr_reas, x2(i,:), 'aerobic_glucose',tol_vec(tol_cnt));
        res_vec(i, tol_cnt) = concentration_perturb(74,end);
        Error_vec(i, tol_cnt) = Error;
        conc_res{end+1} = concentration_perturb;        
    end
    conc_file = strcat('./temp/',num2str(i),'conc_res.mat');
    Error_file = strcat('./temp/',num2str(i),'Error_vec.mat');
    res_file = strcat('./temp/',num2str(i),'res_vec.mat');
    
    save(conc_file,'conc_res');
    save(Error_file,'Error_vec');
    save(res_file,'Error_vec');

end

        
    


