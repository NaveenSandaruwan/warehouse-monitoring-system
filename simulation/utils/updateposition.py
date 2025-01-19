import requests

def update_user_location(wid, x,y):
    current_location = f"({x},{y})"

    url = f"http://localhost:5000/users/wid/{wid}/location"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "current_location": current_location
    }
    response = requests.put(url, json=data, headers=headers)
    return response.json()

# Example usage
if __name__ == "__main__":
    wid = 110
    current_location = (20, 3)
    response = update_user_location(wid, current_location[0], current_location[1])
    print(response)