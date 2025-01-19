import requests

def send_work_data(wid, date, work_done):
    url = "http://localhost:5000/works"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "wid": wid,
        "date": date,
        "work_done": work_done
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()