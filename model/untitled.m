a = -0.2;
b = 0.2;
r = (b-a).*rand(5,1) + a;
succ = zeros(20,1);
Err = zeros(20,1);
prs_mat = zeros(20,5);
ode = zeros(20,1);
r = (b-a).*rand(20,5) + a;
for i = 1:20
    %r = (b-a).*rand(5,1) + a;
    prs_org = [3,10,2,4,0.3];
    prs = prs_org +r(i,:);
    disp(prs)
    for k = 1:numel(prs)
        if prs(k) > 10
            prs(k) = 10;
        elseif prs(k)<0
            prs(k) = 0;
        end
       
    end
    disp(prs)
    prs_mat(i,:) = prs;
    [v,c,E]=Main_Module( [3,54,56,75,82],prs,'aerobic_glucose',5000);
    [row,col]= size(c);
    succ(i)= c(74,end);
    Err(i) = E; 
    ode(i) = col;
end

