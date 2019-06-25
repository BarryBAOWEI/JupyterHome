#!/usr/bin/env python
# coding: utf-8

import requests
import re
import os

def downloads_txt(year,real_path):
    
    year = str(year)
    real_path = real_path
    
    if not os.path.exists(real_path):
        os.makedirs(real_path)
    print('reading...')
    with open(real_path+'txt_url_'+year+'.txt', 'r', encoding='utf-8') as f:
        urls = [x for x in f.read().split('\n') if x]
    print('read success!')
    for url in urls:
        while True:
            try:
                print('正在获取{}'.format(url))
                r = requests.get(url, timeout=120)
                print('获取成功,正在写入')
                if not os.path.exists(real_path+year):
                    os.makedirs(real_path+year)
                with open(real_path+year+'/{}'.format(url.split('/')[-1]), 'w', encoding='utf-8') as f:
                    
                    rtext = re.findall('>(.*?)<',r.text)
                    ################## 删选 ##################
                    rtext_ = []
                    for each in rtext:

                        # 先判断章节是否出现，出现则保存，不出现再另做删除判断
                        if 'Item' in each:
                            rtext_.append(each)
                        else:
                            # 删除年报文本中的短文本 - 乱码、数据、数据表头内容
                            if len(re.findall(' ',each)) <= 5 or '&' in each or 'http' in each:
                                pass
                            else:
                                rtext_.append(each)
                    ##########################################
                    for each in rtext_:
                        f.write(each+'\n')
                break
            except Exception as e:
                print(e, 'ERROR')


if __name__ == '__main__':
    # 总文件路径，该路径下保存有 txt_url_+年份.txt 文件即可 - 其中每行是SEC年报网页源代码txt下载地址
    real_path = 'C:/Users/jxjsj/Desktop/secreport/'
    for year in range(2018,2019):
        year = year
        print(year,'Start')
        downloads_txt(year,real_path)
        print(year,'End')
