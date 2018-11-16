import requests, re
import pandas as pd
import numpy as np
  
wb = pd.read_excel('C:/Users/Administrator/Desktop/reflect.xlsx')
path = 'C:/Users/Administrator/Desktop/新建文件夹 (2)'
lst = np.array(wb).T[0][:500]

record_lst = []
record_lst_1 = []
record_lst_2 = []
save = pd.DataFrame()
save_1 = pd.DataFrame()
save_2 = pd.DataFrame()
cnt = 0
cnt_1 = 0
cnt_2 = 0

for i in lst:
    i_str=str(i)
    url = 'http://fund.eastmoney.com/f10/jjfl_'+i_str+'.html'

    response = requests.get(url)
    response_lst = list(re.findall('''申购费率.*<td class="th">小于100万元</td>.*<strike class='gray'>(.*?)<\/strike>.*<td class="th">大于等于100万元，小于500万元</td>.*<strike class='gray'>(.*?)<\/strike>.*''', response.text))
    if response_lst == []:
        record_lst.append(i_str)
        continue
    print(response_lst)
    w, x, = response_lst[0][0], response_lst[0][1],
    save[cnt] = [i,w,x]
    cnt = cnt + 1
output = save.T
print(output)
if len(output.columns) == 0:
    pass
else:
    output.columns = ['code','小于100万元','大于等于100万元，小于500万元']
    output.to_excel(path+'FeeRateCollect4.xlsx', sheet_name='FeeRate', index=False)

for i in record_lst:
    url = 'http://fund.eastmoney.com/f10/jjfl_'+i+'.html'

    response = requests.get(url)
    response_lst = list(re.findall('''申购费率.*<td class="th">小于100万元</td>.*<strike class='gray'>(.*?)<\/strike>.*<td class="th">大于等于100万元，小于500万元</td>.*<strike class='gray'>(.*?)<\/strike>.*<td class="th">大于等于500万元，小于1000万元</td>.*<strike class='gray'>(.*?)<\/strike>.*''', response.text))
    if response_lst == []:
        record_lst_1.append(i)
        continue
    print(response_lst)
    x, y, z = response_lst[0][0], response_lst[0][1], response_lst[0][2]
    save_1[cnt_1] = [i,x,y,z]
    cnt_1 = cnt_1 + 1
output_1 = save_1.T
print(output_1)
if len(output_1.columns) == 0:
    pass
else:
    output_1.columns = ['code','小于100万元','大于等于100万元，小于500万元','大于等于500万元，小于1000万元']
    output_1.to_excel(path+'FeeRateCollect5.xlsx', sheet_name='FeeRate', index=False)

for i in record_lst_1:
    url = 'http://fund.eastmoney.com/f10/jjfl_'+i+'.html'

    response = requests.get(url)
    response_lst = list(re.findall('''申购费率.*<td class="th">小于50万元</td>.*<strike class='gray'>(.*?)<\/strike>.*<td class="th">大于等于50万元，小于200万元</td>.*<strike class='gray'>(.*?)<\/strike>.*<td class="th">大于等于200万元，小于500万元</td>.*<strike class='gray'>(.*?)<\/strike>.*''', response.text))
    if response_lst == []:
        record_lst_2.append(i)
        continue
    print(response_lst)
    x, y, z = response_lst[0][0], response_lst[0][1], response_lst[0][2]
    save_2[cnt_2] = [i,x,y,z]
    cnt_2 = cnt_2 + 1
output_2 = save_2.T
print(output_2)
if len(output_2.columns) == 0:
    pass
else:
    output_2.columns = ['code','小于50万元','大于等于50万元，小于200万元','大于等于200万元，小于500万元']
    output_2.to_excel(path+'FeeRateCollect6.xlsx', sheet_name='FeeRate', index=False)
print('共',len(lst),'家','爬取',len(record_lst_2),'家')


# output = save.T
# output.columns = ['code','小于50万元','大于等于50万元，小于100万元','大于等于100万元，小于500万元']
# output.to_excel('C:/Users/Administrator/Desktop/FeeRateCollect.xlsx', sheet_name='FeeRate', index=False)

# 申购费率.*<strike class='gray'>(.*?)<\/strike>.*<strike class='gray'>(.*?)<\/strike>.*<strike class='gray'>(.*?)<\/strike>.*<strike class='gray'>(.*?)<\/strike>
# 申购费率.*<strike class='gray'>(.*?)<\/strike>.*<strike class='gray'>(.*?)<\/strike>.*<strike class='gray'>(.*?)<\/strike>