import requests
import json

def search_ebay_sold_items(keywords):
    url = "https://ebay-sold-items-api.herokuapp.com/findCompletedItems"
    headers = {
        'Content-Type': 'application/json',
        'x-rapidapi-key': '2ca9e23269mshd30f3992cdfb978p1f841ejsnb17d1dcff8c8',  # Replace with your actual RapidAPI key
        'x-rapidapi-host': 'ebay-average-selling-price.p.rapidapi.com'
    }
    payload = json.dumps({
        "keywords": keywords,
        "max_search_results": 100,  # You can adjust this number based on your needs
        "remove_outliers": False  # Set to True if you want to remove outliers
    })

    response = requests.post(url, headers=headers, data=payload)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Failed to retrieve data")
        return None

def calculate_average_price(data):
    if not data or 'products' not in data:
        print("No data to calculate average price")
        return

    total_price = 0
    for item in data['products']:
        total_price += item['sale_price']

    average_price = total_price / len(data['products'])
    print(f"Average Price: {average_price}")

# Replace 'iPhone' with your search query
data = search_ebay_sold_items('2015 Equinox hood')
calculate_average_price(data) 
