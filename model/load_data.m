function  load_data()
succ_bio=zeros(2000,2);
for i=1:2000% i= 1
    filename_pr =  ['/home/maryam/Desktop/Data_gen/',num2str(i),'concentration_perturb.mat'];
    conc = load(filename_pr);
    succ_bio(i,1) = conc.concentration_perturb(73,end);
    succ_bio(i,2) = conc.concentration_perturb(111,end);
end

succ_bio(find(succ_bio(:,2)>25),:) = []
