% ������Ԥ��,����Ϊt��,��t+1������δ֪,���Ԥ��һ��:��ΪXt+1δ֪,�����ܵõ�yt+1Ҳ�޷��õ�yt+2��
netc = removedelay(results.net);
inputSeries1 = tonndata(X(end-10:end,:).',true,false);  %tonndata(input_test,true,false)
targetSeries1 = tonndata(y(end-10:end,:).',true,false); %tonndata(output_tests,true,false)
%����delay���õ�3,�����������Xn,����������Yn,ϣ��Ԥ��δ֪��Yn+1
%input_test=[Xn-2,Xn-1,Xn],output_tests=[Yn-2,Yn-1,Yn],����"��Ϊ����&��Ϊ�۲�"
[xs,xis,ais,ts] = preparets(netc,inputSeries1,{},targetSeries1); 
y_predict = netc(xs,xis,ais); 


netc = removedelay(results.net);
targetSeries1 = tonndata(y(end-10:end,:).',true,false); %tonndata(output_tests,true,false)
%����delay���õ�3,����������Yn,ϣ��Ԥ��δ֪��Yn+1
%output_tests=[Yn-2,Yn-1,Yn],����"��Ϊ����&��Ϊ�۲�"
[xs,xis,ais,ts] = preparets(netc,{},{},targetSeries1); 
y_predict = netc(xs,xis,ais); 