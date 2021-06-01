function aut_plots(c)
%x1 = [39, 11, 407, 33,25,26,27,28,29,30,31,32,187,124,377];
%x2=repmat(50,1,length(x1));
%x2(13:15)=[0.1,0.1,0.1];
x=1:337;
%x2=10;
for i =1:length(x)
    %[c]=main_module(x1(i),x2, 'aerobic_glucose');
    plot(c(x(i),:))
    xlabel('time')
    ylabel('EX-glc(e) flux value')
    title(num2str(x(i)))
    filename = ['/home/maryam/Desktop/plots/exploratory/' num2str(x(i)) '.jpg'];
    saveas(1,filename)
    %plot(v(83,:)./v(25,:))
    %xlabel('time')
    %ylabel('EX-succ(e) flux value')
    %title(num2str(x1(i)))
    %filename2 = ['/home/maryam/Desktop/desk3/model/explore_model/rand/succ-' num2str(x1(i)) '.jpg'];
    %saveas(1,filename2)
   
 
end



