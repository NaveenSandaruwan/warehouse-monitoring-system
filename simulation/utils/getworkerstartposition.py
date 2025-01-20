import requests

def get_user_location(wid):
    url = f"http://localhost:5000/users/wid/{wid}/location"
    headers = {
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        current_location = data.get("current_location")
        if current_location:
            # Convert the string "(x,y)" to a tuple of integers (x, y)
            x, y = map(int, current_location.strip("()").split(","))
            return (x, y)
    return None

# Example usage
if __name__ == "__main__":
    wid = 111
    location = get_user_location(wid)
    print(location)  # Output should be a tuple of integers, e.g., (20, 3)