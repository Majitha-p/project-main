from django.shortcuts import render, redirect, reverse
from .car import w_calc, prepare_X
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup


w_0, w = w_calc()


def predict(request):
    dataset_path = r'C:\Users\hp\OneDrive\Desktop\car-p\Django-car-price-predictore\main\data.csv'  # Update with the path to your dataset
    df = pd.read_csv(dataset_path)

    # Extract unique car makes from the dataset
    car_makes = df['Make'].unique()
    

    if request.method == 'POST':
        df = {
            'make': request.POST['make'],
            'model': request.POST['model'],
            'year': int(request.POST['year']),
            'engine_fuel_type': request.POST['engine_fuel_type'],
            'engine_hp': int(request.POST['engine_hp']),
            'engine_cylinders': int(request.POST['engine_cylinders']),
            'transmission_type': request.POST['transmission_type'],
            'driven_wheels': request.POST['driven_wheels'],
            'number_of_doors': int(request.POST['number_of_doors']),
            'market_category': request.POST['market_category'],
            'vehicle_size': request.POST['vehicle_size'],
            'vehicle_style': request.POST['vehicle_style'],
            'highway_mpg': int(request.POST['highway_mpg']),
            'city_mpg': int(request.POST['city_mpg']),
            'popularity': int(request.POST['popularity']),
        }

        X_test = prepare_X(pd.DataFrame([df]))
        y_pred = w_0 + X_test.dot(w)
        price = np.expm1(y_pred)[0].astype(int)

        # Scrape Cars.com to get additional details
        url = reverse('scrape') + f"?make={df['make']}&model={df['model']}&year={df['year']}&price={price}"

        # Redirect to the scrape view with constructed URL
        return redirect(url)
    else:
        return render(request, 'index.html', {'car_makes': car_makes})
    
   



 
# def scrape_cars(make, model, year):
#     make_param = f"makes[]={make}"
#     model_params = [f"models[]={m}" for m in model]
#     model_param_string = "&".join(model_params)
#     url = f"https://www.cars.com/shopping/results/?{make_param}"
#     response = requests.get(url)
#     if response.status_code == 200:
#         soup = BeautifulSoup(response.content, 'html.parser')
#         listings = soup.find_all('div', class_='vehicle-card')
#         print(listings)
#         car_details = []
#         for listing in listings:
#             details = {}
#             details['make'] = make
#             details['model'] = model
#             details['year'] = year
            
#             # Check if the price element exists
#             # price_element = listing.find('span', class_='listing-row__price')
#             # details['price'] = price_element.text.strip() if price_element else "Price not available"
            
#             # Similarly, handle other elements
            
#             mileage_element = listing.find('div', class_='mileage')
#             details['mileage'] = mileage_element.text.strip() if mileage_element else "Mileage not available"
            
#             location_element = listing.find('div', class_='dealer-name')
#             details['location'] = location_element.text.strip() if location_element else "Location not available"
            
#             car_details.append(details)
#         return car_details
#     else:
#         print("Failed to fetch data from Cars.com")
#         return []


def scrape(request):
    # Retrieve entered car details from the previous page
    make = request.GET.get('make')
    model = request.GET.get('model')
    year = request.GET.get('year')
    price = request.GET.get('price')
    
    # Construct the URL for scraping results
    url = reverse('scrape') + f"?make={make}&model={model}&year={year}&price={price}"
    
    # Render the scrape.html template with the car details and URL
    return render(request, 'scrape.html', {'make': make, 'model': model, 'year': year, 'price': price, 'url': url})
