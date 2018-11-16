import requests, re
import pandas as pd
import numpy as np

#######读写路径 必改######
wb = pd.read_excel('C:/Users/jxjsj/Desktop/reflect.xlsx')
path = 'C:/Users/jxjsj/Desktop/'
lst = np.array(wb).T[0]

####批量执行各匹配种类####
record_lst_0 = [str(i).zfill(6) for i in lst]
record_lst_1 = []
record_lst_2 = []
record_lst_3 = []
##########################

##记录未匹配任何内容代码##
record_lst_4 = []
##########################

########保存结果表########
save_0 = pd.DataFrame()
save_1 = pd.DataFrame()
save_2 = pd.DataFrame()
save_3 = pd.DataFrame()
##########################

##########计数器##########
cnt_0 = 0
cnt_1 = 0
cnt_2 = 0
cnt_3 = 0
##########################


###########1号############
for i in record_lst_0:
    print(i)
    url_0 = 'http://fund.eastmoney.com/f10/jjfl_'+i+'.html'

    response_0 = requests.get(url_0)
    response_lst_0 = list(re.findall("申购费率.*<strike class='gray'>(.*?)<\/strike>.*<strike class='gray'>(.*?)<\/strike>.*<strike class='gray'>(.*?)<\/strike>.*<strike class='gray'>(.*?)<\/strike>", response_0.text))
    if response_lst_0 == []:
        record_lst_1.append(i)
        continue
    print(response_lst_0)
    w, x, y, z = response_lst_0[0][0], response_lst_0[0][1], response_lst_0[0][2], response_lst_0[0][3]
    save_0[cnt_0] = [i,w,x,y,z]
    cnt_0 = cnt_0 + 1
output_0 = save_0.T
if len(output_0.columns) == 0:
    pass
else:
    output_0.columns = ['code','小于50万元','大于等于50万元，小于100万元','大于等于100万元，小于300万元','大于等于300万元，小于500万元']
    output_0.to_excel(path+'FeeRateCollect1.xlsx', sheet_name='FeeRate', index=False)
##########################


###########2号############
for ii in record_lst_1:
    print(ii)
    url_1 = 'http://fund.eastmoney.com/f10/jjfl_'+ii+'.html'

    response_1 = requests.get(url_1)
    response_lst_1 = list(re.findall("申购费率.*<strike class='gray'>(.*?)<\/strike>.*<strike class='gray'>(.*?)<\/strike>.*<strike class='gray'>(.*?)<\/strike>", response_1.text))
    if response_lst_1 == []:
        record_lst_2.append(i)
        continue
    print(response_lst_1)
    x, y, z = response_lst_1[0][0], response_lst_1[0][1], response_lst_1[0][2]
    save_1[cnt_1] = [ii,x,y,z]
    cnt_1 = cnt_1 + 1
output_1 = save_1.T
print(output_1)
if len(output_1.columns) == 0:
    pass
else:
    output_1.columns = ['code','小于50万元','大于等于50万元，小于100万元','大于等于100万元，小于500万元']
    output_1.to_excel(path+'FeeRateCollect2.xlsx', sheet_name='FeeRate', index=False)
##########################


###########3号############
for iii in record_lst_2:
    print(iii)
    url_2 = 'http://fund.eastmoney.com/f10/jjfl_'+iii+'.html'

    response_2 = requests.get(url_2)
    response_lst_2 = list(re.findall('''申购费率.*<\/td><td>(.*?)<\/td><\/tr><tr>.*<\/td><td>(.*?)<\/td><\/tr><tr>.*<\/td><td>(.*?)<\/td><\/tr><tr>.*<td class="th">大于等于500万元''', response_2.text))
    if response_lst_2 == []:
        record_lst_3.append(i)
        continue
    print(response_lst_2)
    x, y, z = response_lst_2[0][0], response_lst_2[0][1], response_lst_2[0][2]
    save_2[cnt_2] = [iii,x,y,z]
    cnt_2 = cnt_2 + 1
output_2 = save_2.T
print(output_2)
if len(output_2.columns) == 0:
    pass
else:
    output_2.columns = ['code','小于100万元','大于等于100万元，小于200万元','大于等于200万元，小于500万元']
    output_2.to_excel(path+'FeeRateCollect3.xlsx', sheet_name='FeeRate', index=False)
##########################


###########4号############
for iiii in record_lst_3:
    print(iiii)
    url_3 = 'http://fund.eastmoney.com/f10/jjfl_'+iiii+'.html'

    response_3 = requests.get(url_3)
    response_lst_3 = list(re.findall("申购费率.*<strike class='gray'>(.*?)<\/strike>.*<strike class='gray'>(.*?)<\/strike>", response_3.text))
    if response_lst_3 == []:
        record_lst_4.append(i)
        continue
    print(response_lst_3)
    x, y = response_lst_3[0][0], response_lst_3[0][1]
    save_3[cnt_3] = [iiii,x,y]
    cnt_3 = cnt_3 + 1
output_3 = save_3.T
if len(output_3.columns) == 0:
    pass
else:
    output_3.columns = ['code','小于100万元','大于等于100万元，小于500万元']
    output_3.to_excel(path+'FeeRateCollect4.xlsx', sheet_name='FeeRate', index=False)
##########################


##########计数############
print('共',len(lst),'家','爬取',len(lst) - len(record_lst_4),'家')
##########################


########未爬取代码########
no_save = pd.DataFrame(np.array(record_lst_4).T)
no_save.columns = ['无匹配']
no_save.to_excel(path+'NoCollection.xlsx', sheet_name='oh_no_!', index=False)
##########################


#####新添匹配格式 模板####
# for i in record_lst_【】:
    # url = 'http://fund.eastmoney.com/f10/jjfl_'+i+'.html'

    # response = requests.get(url)
    # response_lst = list(re.findall(正则语句, response.text))
    # if response_lst == []:
        # record_lst_【+1】.append(i)
        # continue
    # print(response_lst)
    # x, y, z = response_lst[0][0], response_lst[0][1], response_lst[0][2]
    # save_【】[cnt] = [i,x,y,z]
    # cnt_【】 = cnt_【】 + 1
# output_【】 = save_【】.T
# print(output_【】)
# if len(output_【】.columns) == 0:
    # pass
# else:
    # output_【】.columns = ['code','小于100万元','大于等于100万元，小于200万元','大于等于200万元，小于500万元']
    # output_【】.to_excel(path+'FeeRateCollect999.xlsx', sheet_name='FeeRate', index=False)
##########################





# output = save.T
# output.columns = ['code','小于50万元','大于等于50万元，小于100万元','大于等于100万元，小于500万元']
# output.to_excel('C:/Users/jxjsj/Desktop/FeeRateCollect.xlsx', sheet_name='FeeRate', index=False)

# 申购费率.*<strike class='gray'>(.*?)<\/strike>.*<strike class='gray'>(.*?)<\/strike>.*<strike class='gray'>(.*?)<\/strike>.*<strike class='gray'>(.*?)<\/strike>
# 申购费率.*<strike class='gray'>(.*?)<\/strike>.*<strike class='gray'>(.*?)<\/strike>.*<strike class='gray'>(.*?)<\/strike>