import folium
m = folium.Map(location=[35.15321829786845,128.09976362791986],zoom_start=20)

folium.Marker([35.15321829786845,128.09976362791986],popup="<h1> 집가고싶다 </h1>",tooltip="나의 위치",icon=folium.Icon(icon="heart",icon_color="red")).add_to(m)

m.save('map.html')
