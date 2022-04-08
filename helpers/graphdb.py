import requests

def load_graph():
    url = "http://MacBook-Pro-van-Tiddo.local:7200/repositories/graphs"
    headers = {'Content-Type': 'application/x-turtle'}
    with open('../data/AIFB/AIFB_complete.n3', 'rb') as f:
        r = requests.post(url, headers=headers, files={'file':f})
    print(r.status_code)

if __name__=='__main__':
    load_graph()