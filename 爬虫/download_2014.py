import requests
import re
import os
import threading


def downloads_htm():
    if not os.path.exists('data'):
        os.makedirs('data')
    for year in range(2014, 2015):
        with open('txt_url_{}.txt'.format(str(year)), 'w', encoding='utf-8') as f:
            if not os.path.exists('data/{}'.format(str(year))):
                os.makedirs('data/{}'.format(str(year)))
            for qtr in range(1, 5):
                while True:
                    try:
                        url = 'https://www.sec.gov/Archives/edgar/full-index/{0}/QTR{1}/company.idx'.format(str(year), str(qtr))
                        print('正在获取{}___{}'.format(str(year), str(qtr)), end='\t')
                        r = requests.get(url, timeout=120)
                        print('获取成功')
                        break
                    except:
                        print('ERROR')
                for line in r.text.split('\n'):
                    if ('10-K' in line) and ('10-K/A' not in line) and ('NT 10-K' not in line) and ('10-KT/A' not in line) \
                            and ('10-KT' not in line):
                        url_htm = 'https://www.sec.gov/Archives/'+re.search(r'edgar/.+?\.txt', line).group()
                        f.write(url_htm+'\n')


# downloads_htm()

def downloads_txt():
    if not os.path.exists('data'):
        os.makedirs('data')
    print('reading...')
    with open('txt_url_2014.txt', 'r', encoding='utf-8') as f:
        urls = [x for x in f.read().split('\n') if x]
    print('read success!')
    for url in urls:
        while True:
            try:
                print('正在获取{}'.format(url))
                r = requests.get(url, timeout=120)
                print('获取成功,正在写入')
                with open('2014/{}'.format(url.split('/')[-1]), 'w', encoding='utf-8') as f:
                    f.write(r.text)
                break
            except Exception as e:
                print(e, 'ERROR')

downloads_txt()
