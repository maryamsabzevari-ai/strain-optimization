function [Vnet,concentration_perturb]= Data_gen()
iterations = 10^6;
pr_num = 5;
%a = 1;
%b = 337;
%r = round((b-a).*rand(1000000000,5) + a);
s = RandStream('mlfg6331_64'); 
%y = datasample(s,1:457,5,'Replace',false);
pr_vals = [0:0.1:1 2:10];
%size(r,1)
x1 = zeros(iterations,pr_num);
x2 = zeros(iterations,pr_num);
for i=1:iterations
    x1(i,:) = datasample(s,1:457,5,'Replace',false);
    x2(i,:) = datasample(s,pr_vals,5,'Replace',true);
    [Vnet,concentration_perturb]=Main_Module(x1(i,:),x2(i,:),'aerobic_glucose');
    vnet_fieldnames = strcat('./Data_gen/th',num2str(i),'Vnet.mat');
    conc_fieldnames = strcat('./Data_gen/th',num2str(i),'concentration_perturb.mat');
    save(vnet_fieldnames,'Vnet');
    save(conc_fieldnames,'concentration_perturb');
   
    reactions = x1(i,:);
    perturbations = x2(i,:);
    
    x1_file = strcat('./Data_gen/th',num2str(i),'reactions.mat');
    x2_file = strcat('./Data_gen/th',num2str(i),'perturbations.mat');
    
    save(x1_file,'reactions');
    save(x2_file,'perturbations');
 
end



    
