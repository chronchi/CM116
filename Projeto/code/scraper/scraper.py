from bs4 import BeautifulSoup
import requests
import csv

html = requests.get("https://www.vivareal.com.br/venda/parana/curitiba/apartamento_residencial/").text
soup = BeautifulSoup(html, 'html5lib')

price = soup('div', 'property-card__price')

uls = soup("ul", "property-card__details")

room2 = soup("span","property-card__detail-value property-card__detail-area")

rooms2 = []

for i in range(0,len(room2)):
    rooms2.append(room2[i].text.strip().encode("utf-8"))

with open("num_room.csv","wb") as file:
    writer = csv.writer(file)
    for i in range(0,len(rooms2)):
        writer.writerow([rooms2[i]])

prices = []
for i in range(0,len(price)):
    prices.append(price[i].text.strip().strip("R$ ").encode("utf-8"))

with open("price.csv", "wb") as file:
    writer = csv.writer(file)
    for i in range(0,len(prices)):
        writer.writerow([prices[i]])
