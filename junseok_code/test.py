from flask import Flask, render_template_string
import folium

app = Flask(__name__)

@app.route("/")
def index():
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Current Location</title>
        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"/>
    </head>
    <body>
        <div id="map" style="width: 100%; height: 600px;"></div>
        <script>
            var map = L.map('map').setView([0, 0], 20);

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }).addTo(map);

            var current_position, current_accuracy;

            function onLocationFound(e) {
                if (e.accuracy > 100) { // Adjust this value to set the desired accuracy
                return;
                }

                if (current_position) {
                map.removeLayer(current_position);
                map.removeLayer(current_accuracy);
                }

                var radius = e.accuracy / 2;

                current_position = L.marker(e.latlng).addTo(map)
                .bindPopup("You are within " + radius + " meters from this point").openPopup();

                current_accuracy = L.circle(e.latlng, radius).addTo(map);
                }

            function onLocationError(e) {
                alert(e.message);
            }

            function updatePosition() {
                map.locate({setView: false, maxZoom: 16, enableHighAccuracy: true, timeout: 10000});
            }

            map.on('locationfound', onLocationFound);
            map.on('locationerror', onLocationError);

            // Update position every 10 seconds
            setInterval(updatePosition, 10000);
            updatePosition();
        </script>
    </body>
    </html>
    '''

    return render_template_string(html_template)

if __name__ == "__main__":
    app.run(debug=True)