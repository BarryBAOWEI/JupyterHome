import requests, re

url = 'http://fund.eastmoney.com/f10/jjfl_502048.html'
response = requests.get(url)
response_lst = re.findall("申购费率（前端）.*小于50万元.*'gray'>(.*?)<\/strike>.*大于等于50万元，小于100万元.*'gray'>(.*?)<\/strike>.*大于等于100万元，小于500万元.*'gray'>(.*?)<\/strike>", response.text)

print(response_lst)