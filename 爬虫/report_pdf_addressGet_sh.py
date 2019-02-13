import requests
import math
import csv
import os
import re

# --正式开始爬虫--

def getAlldownloadAddress(year_list, industry_list, out_put_file, error_file):
    '''
    year_list     : 年份（字符串格式）的列表，如['2010','2011','2012','2013']
    industry_list : 行业（字符串格式）的列表，取自
                                               (农、林、牧、渔业;
                                                采矿业;
                                                制造业;
                                                电力、热力、燃气及水生产和供应业;
                                                建筑业;
                                                批发和零售业;
                                                交通运输、仓储和邮政业;
                                                住宿和餐饮业;
                                                信息传输、软件和信息技术服务业;
                                                房地产业;
                                                租赁和商务服务业;
                                                科学研究和技术服务业;
                                                居民服务、修理和其他服务业;
                                                教育;
                                                卫生和社会工作;
                                                文化、体育和娱乐业;
                                                综合;)
    out_put_file : 输出内容存放的文件夹目录
    '''
    
    # 基本参数 - 固定
    """
    *****************
    固定配置，勿改
    *****************
    """
    URL = 'http://www.cninfo.com.cn/new/hisAnnouncement/query'
    HEADER = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest'
    }
    MAX_PAGESIZE = 50
    MAX_RELOAD_TIMES = 5
    RESPONSE_TIMEOUT = 10
    
    for year_each in year_list:
        
        for industry_each in industry_list:
            
            # 基本参数 - 可改
            """
            *****************
            可改配置，酌情更改
            *****************
            """
            announcementYear = year_each
            seDate = announcementYear+'-01-01~'+announcementYear+'-12-31'
            trade = industry_each

            # 第一次请求获得总条目数，用以计算页数
            query = {
                'pageNum': ''
                ,'pageSize': MAX_PAGESIZE
                ,'tabName': 'fulltext'
                ,'column': 'sse'
                ,'stock': ''
                ,'searchkey': ''
                ,'secid': ''
                ,'plate': 'shmb'
                ,'category': 'category_ndbg_szsh;'
                ,'trade': trade
                ,'seDate': seDate
            }
            r = requests.post(URL, query, HEADER, timeout=RESPONSE_TIMEOUT)
            my_query = r.json()
            total_page_num = math.ceil(my_query['totalAnnouncement']/MAX_PAGESIZE)

            #  逐页抓取
            result_list = []
            for pageNum in range(1,total_page_num+1):
                query = {
                'pageNum': pageNum
                ,'pageSize': MAX_PAGESIZE
                ,'tabName': 'fulltext'
                ,'column': 'sse'
                ,'stock': ''
                ,'searchkey': ''
                ,'secid': ''
                ,'plate': 'shmb'
                ,'category': 'category_ndbg_szsh;'
                ,'trade': trade
                ,'seDate': seDate
                }
            #     r = requests.post(URL, query, HEADER, timeout=RESPONSE_TIMEOUT)

                reloading = 0
                while True:
                    reloading += 1
                    if reloading > MAX_RELOAD_TIMES:
                        break
            #         elif reloading > 1:
            #             __sleeping(random.randint(5, 10))
            #             print('... reloading: the ' + str(reloading) + ' round ...')
                    try:
                        r = requests.post(URL, query, HEADER, timeout=RESPONSE_TIMEOUT)
                    except Exception as e:
                        print(e)
                        continue
                    if r.status_code == requests.codes.ok and r.text != '':
                        break
                my_query = r.json()
                r.close()

                for each in my_query['announcements']:
                    file_link = 'http://www.cninfo.com.cn/' + str(each['adjunctUrl'])
                    file_name = str(each['secCode']) + str(each['secName']) + str(each['announcementTitle']) + file_link[-4:]
                    result_list.append([file_name, file_link])
                
                print('Page%d/%d'% (pageNum,total_page_num),announcementYear,trade )
        
            out_put_file = out_put_file

            for result in result_list:
                
                try:
                    year = re.findall('(20\d+)年',result[0])[0]
                except:
                    print(result)
                    errorAddress = open(error_file+'errorAddress_SH.txt', 'a')
                    for i in result:
                        errorAddress.write(i)
                    errorAddress.write('\r\n')
                    errorAddress.close()

                if os.path.exists(out_put_file+str(year)):
                    pass
                else:
                    os.makedirs(out_put_file+str(year))

                output_csv_file = out_put_file+str(year)+'/'+trade+'.csv'
                with open(output_csv_file, 'a', newline='', encoding='gb18030') as csv_out:
                    writer = csv.writer(csv_out)
                    writer.writerows([result])
            print(announcementYear,trade)

####### RUN ################################################################################################
            
getAlldownloadAddress(year_list     = [str(2014+i) for i in range(1)], 
                      industry_list = [
                                        '农、林、牧、渔业',
                                        '采矿业',
                                        '制造业',
                                        '电力、热力、燃气及水生产和供应业',
                                        '建筑业',
                                        '批发和零售业',
                                        '交通运输、仓储和邮政业',
                                        '住宿和餐饮业',
                                        '信息传输、软件和信息技术服务业',
                                        '房地产业',
                                        '租赁和商务服务业',
                                        '科学研究和技术服务业',
                                        '居民服务、修理和其他服务业',
                                        '教育',
                                        '卫生和社会工作',
                                        '文化、体育和娱乐业',
                                        '综合'
                                        ], 
                      out_put_file  = 'D:/report_pdf_sh/',
                      error_file = 'D:/report_pdf_error/') # 自行创建