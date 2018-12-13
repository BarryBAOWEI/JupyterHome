% 真正的预测,今天为t天,对t+1天数据未知,最多预测一天:因为Xt+1未知,即便能得到yt+1也无法得到yt+2！
netc = removedelay(results.net);
inputSeries1 = tonndata(X(end-10:end,:).',true,false);  %tonndata(input_test,true,false)
targetSeries1 = tonndata(y(end-10:end,:).',true,false); %tonndata(output_tests,true,false)
%假设delay设置的3,最近的输入是Xn,最近的输出是Yn,希望预测未知的Yn+1
%input_test=[Xn-2,Xn-1,Xn],output_tests=[Yn-2,Yn-1,Yn],都是"行为变量&列为观测"
[xs,xis,ais,ts] = preparets(netc,inputSeries1,{},targetSeries1); 
y_predict = netc(xs,xis,ais); 


netc = removedelay(results.net);
targetSeries1 = tonndata(y(end-10:end,:).',true,false); %tonndata(output_tests,true,false)
%假设delay设置的3,最近的输出是Yn,希望预测未知的Yn+1
%output_tests=[Yn-2,Yn-1,Yn],都是"行为变量&列为观测"
[xs,xis,ais,ts] = preparets(netc,{},{},targetSeries1); 
y_predict = netc(xs,xis,ais); 