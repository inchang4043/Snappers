import geocoder
import folium

g = geocoder.ip('me')
lat, lng = g.latlng

map = folium.Map(location=[lat, lng], zoom_start=13)

folium.Marker(location=[lat, lng], popup="My Location").add_to(map)

map.save("my_location2.html")
