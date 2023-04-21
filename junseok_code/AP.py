import requests
import json
import folium

response = requests.get("https://geolocation-db.com/json/")
location = json.loads(response.content.decode())

lat, lng = location["latitude"], location["longitude"]

map = folium.Map(location=[lat, lng], zoom_start=13)

folium.Marker(location=[lat, lng], popup="My Location").add_to(map)

map.save("my_location.html")
