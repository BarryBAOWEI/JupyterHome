#!/usr/bin/env python
# coding: utf-8
from lxml import etree
import requests
import re
import glob


def select_listed_company(company_txt):
    """
    筛选非New York Stock Exchange和NASDAQ上市的公司
    对文本前2000个字符做俩个判断:
    1.文本是否包含
    name of each exchange on which registered
    or
    name of exchange on which registered
    这俩个关键词
    2.是否包含
    New York Stock Exchange
    or
    NASDAQ
    关键词
    :return True or False
    """
    if ('name of each exchange on which registered' in company_txt) or (
            'name of exchange on which registered' in company_txt) or (
            'New York Stock Exchange' in company_txt) or (
            'NASDAQ' in company_txt) or (
            'NYSE' in company_txt
    ):
        return True
    else:
        return False


def get_text_from_url(url_):
    """
    from url get text from disk
    :param url_:
    :return:
    """
    path = 'data/2014/2014'
    with open(path+url_.split('/')[-1], 'r', encoding='utf-8') as f:
        return f.read()


def get_central_index_key(url_):
    """
    获取 central index key 一般在文件名前面那段数字
    :param url_: https://www.sec.gov/Archives/edgar/data/1141807/0001214659-14-002350.txt
    :return:
    """
    return url_.split('/')[-1].split('-')[0]


def get_year(url_):
    """
    get the year, always in the url
    :param url_:
    :return: str
    """
    return url_.split('/')[-1].split('-')[1]


def get_company_txt(url_):
    """
    根据链接获取文章内容
    :param url_:
    :return:
    """
    while True:
        try:
            r = requests.get(url_)
            return r.text
        except Exception as e:
            print(e)
            print('ERROR')


def get_ticker_symbol(text_):
    """
    get company ticker symbol
    often show in "under the symbol" or "under the stock symbol" or "under the ticker symbol“
        "with the ticker symbol" or "with the symbol" or "with the stock symbol" after
    :param text_:the html
    :return: str
    """
    if re.findall(r'under the symbol &#8220;(.+?)&#8221;', text_):
        return re.findall(r'under the symbol &#8220;(.+?)&#8221;', text_)[0].replace('"', '')
    elif re.findall(r'under the symbol (.+?) ', text_):
        return re.findall(r'under the symbol (.+?) ', text_)[0].replace('"', '')
    elif re.findall(r'under the stock symbol (.+?) ', text_):
        return re.findall(r'under the stock symbol (.+?) ', text_)[0].replace('"', '')
    elif re.findall(r'under the ticker symbol (.+?) ', text_):
        return re.findall(r'under the ticker symbol (.+?) ', text_)[0].replace('"', '')
    elif re.findall(r'with the ticker symbol (.+?) ', text_):
        return re.findall(r'with the ticker symbol (.+?) ', text_)[0].replace('"', '')
    elif re.findall(r'with the symbol (.+?) ', text_):
        return re.findall(r'with the symbol (.+?) ', text_)[0].replace('"', '')
    elif re.findall(r'with the stock symbol (.+?) ', text_):
        return re.findall(r'with the stock symbol (.+?) ', text_)[0].replace('"', '')
    else:
        return ''


def get_urls_from_disk(year=2014):
    with open('txt_url_{}.txt'.format(str(year)), 'r', encoding='utf-8') as f:
        return [x for x in f.read().split('\n') if x]


def main():
    urls = get_urls_from_disk()
    for url in urls:
        print(url)
        text = get_text_from_url(url)
        if select_listed_company(text):
            data = list()
            tables = re.findall(r'(<table.+?/table>)', text, re.S)
            for table in tables:
                if 'Deferred Tax Assets' in table:
                    data.append(table)
            if data:
                with open('html/{}.html'.format(url.split('/')[-1]), 'w', encoding='utf-8') as f:
                    f.write('<html>{}</html>'.format("".join(data)))


def test():
    url = '0000004127-14-000046.txt'
    text = get_text_from_url(url)
    print(url)
    # print(text)
    if select_listed_company(text):
        print(url)
        print('year--->{}'.format(get_year(url)))
        print('key----->{}'.format(get_central_index_key(url)))
        print('symbol----->{}'.format(get_ticker_symbol(text)))


if __name__ == '__main__':
    main()
    # test()


