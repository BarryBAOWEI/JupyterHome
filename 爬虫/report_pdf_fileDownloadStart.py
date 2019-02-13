import os
import time

while True:
	try:
		os.system("python C:/Users/jxjsj/Desktop/JupyterHome/爬虫/report_pdf_fileDownload.py")
		break
	except:
		
		#### 可关闭 - 每爬一段sleep一段时间 - 写在 是否已下载 与 是否需要下载 判断之后
		sleep_download_time = 20
		print('ReStart WaitFor%dS' %sleep_download_time)
		time.sleep(sleep_download_time) #这里时间自己设定
		continue