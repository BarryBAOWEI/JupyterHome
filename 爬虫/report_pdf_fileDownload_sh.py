import pandas as pd
import urllib
import os
import socket
import time

def report_pdf_download(previous_save_file, file_save_root, error_save_root, py_file):
    '''
    previous_save_file : 保存有pdf下载地址csv的总目录
    file_save_root     ：用以保存pdf文件的总目录
    '''
    previous_save_file = previous_save_file
    file_save_root = file_save_root
    
    # 固定参数，添加一个请求头
    HEADER = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    RESPONSE_TIMEOUT = 10
    data_header = bytes(urllib.parse.urlencode(HEADER), encoding='utf8') # 二进制编码的请求参数

    total_file = os.listdir(previous_save_file) # 读取第一级目录

    for each_file in total_file:

        downloaded_file = previous_save_file+'/'+each_file
        downloaded_lst = os.listdir(downloaded_file) # 读取第二级目录

        # 是否存在储存pdf用的年份文件夹，不存在则创建
        if os.path.exists(file_save_root+'/'+each_file):
            pass
        else:
            os.makedirs(file_save_root+'/'+each_file)

        for each_csv in downloaded_lst:

            # 是否存在储存pdf用的年份文件夹下的行业文件夹，不存在则创建
            if os.path.exists(file_save_root+'/'+each_file+'/'+each_csv[:-4]):
                pass
            else:
                os.makedirs(file_save_root+'/'+each_file+'/'+each_csv[:-4])

            file_save_pdf = file_save_root+'/'+each_file+'/'+each_csv[:-4] # 用于保存该年份该行业的pdf年报的文件夹

            download_df_file = downloaded_file+'/'+each_csv
            download_df = pd.read_csv(download_df_file, engine='python' , header = None) # 读取包含下载地址的csv文件

            already_downloaded_list = os.listdir(file_save_pdf)

            for i in range(len(download_df)):
                
                stock_pdf = download_df.iloc[i,]

                # 已经下载过则不再下载
                if stock_pdf[0] in already_downloaded_list:
                    continue

                # 标题包含已取消与摘要不下载
                if '已取消' in stock_pdf[0] or '摘要' in stock_pdf[0] or '英文版' in stock_pdf[0]:
                    continue

#                 #### 可关闭 - 每爬一段sleep一段时间 - 写在 是否已下载 与 是否需要下载 判断之后
#                 timeout = 20
#                 socket.setdefaulttimeout(timeout) #这里对整个socket层设置超时时间。后续文件中如果再使用到socket，不必再设置
#                 sleep_download_time = 2
#                 time.sleep(sleep_download_time) #这里时间自己设定
                
                # 报错下载地址记录，暂时好像没起作用？
                try:
#                     u = urllib.request.urlopen(stock_pdf[1]) # 下载文件
                    u = urllib.request.urlopen(stock_pdf[1], data_header, timeout=RESPONSE_TIMEOUT) # 下载文件，改为post形式
                except:
                    errorAddress = open(error_save_root+'/'+'errorAddressDownload_SH.txt', 'a')
                    errorAddress.write(stock_pdf[1]+'\r\n')
                    errorAddress.close()
                    print(stock_pdf[1])
                    continue

                stock_pdf_name = stock_pdf[0].replace('*',"S") # 将*ST改为SST
                f = open(file_save_pdf+'/'+stock_pdf_name, 'wb')

                block_sz = 8192
                
                try:
                    while True:
                        buffer = u.read(block_sz)
                        if not buffer:
                            break
                        f.write(buffer) # 保存文件
                except:
                    print('Restart')
                    sleep_download_time = 20
                    time.sleep(sleep_download_time)
                    os.system("python "+py_file)

                f.close()
                u.close()
                pct = i/len(download_df)
                print(each_file+each_csv[:-4]+'%.3f' % pct, end="\r") # 记录进度

            print(each_file+each_csv[:-4],'finished!')

####### RUN ################################################################################################

report_pdf_download(previous_save_file = 'D:/report_pdf_sh',
                    file_save_root = 'D:/report_pdf_download_sh',
                    error_save_root = 'D:/report_pdf_error',
                    py_file = 'C:/Users/jxjsj/Desktop/JupyterHome/爬虫/report_pdf_fileDownload_sh.py') # 就是你的本py文件的具体路径，注意斜杠方向