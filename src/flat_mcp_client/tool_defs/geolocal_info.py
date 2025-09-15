import requests
from datetime import datetime
#import openmeteo_requests
import requests_cache

from flat_mcp_client.tools import Toolbox
from flat_mcp_client import debug, debug_pp

# TOOL DEFINITIONS
tools  = [
    {
        "type": "function",
        "function": {
            "name": "estimate_gps_coordinates",
            "description": "Approximates GPS coordinates from public IP address",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_nearest_city",
            "description": "Get nearest town/city based on public IP address",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather_foreast_from_gps_coorinates",
            "description": "Get today's hourly weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "float", "description": "Current latitude"},
                    "longitude": {"type": "float", "description": "Current longitude"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]

# CALLABLE PYTHON FUNCTIONS
class GeolocalInfoToolbox(Toolbox):
    """Example class that inherits from flat_mcp_client.Toolbox
    and represents both the tool descriptions (by its inherited initializer)
    and callable functons.
    """

    # class variable to store location results by ip
    cached_ip_results = {}

    # class variable for the openmeteo session
    openmeteocache_session = requests_cache.CachedSession('.cache', expire_after = 3600)

    # and two helper methods
    @staticmethod
    def get_public_IP_address() -> str | None:
        """Get public IP address through free-for-noncomercial use API
        or return None if the API call fails
        """
        url = 'https://api.ipify.org?format=json'
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            return data['ip']
        else:
            return None

    @classmethod
    def get_ip_result(cls) -> dict:
        """Call ipify's API to get location data from
        public IP address, using cached results when possible
        """
        ip = cls.get_public_IP_address()
        if not ip:
            return {"error": "connection error"}
        else:
            # use cached result if available
            data = {}
            if ip in cls.cached_ip_results:
                data = cls.cached_ip_results[ip]
                debug("Retrieved cached result: \n")
                debug_pp(data)
                return data
            else:
                url = f'http://ip-api.com/json/{ip}'
                response = requests.get(url)
                if response.status_code != 200:
                    return {"error": f"Status code: {response.status_code}"}
                else:
                    data = response.json()
                    debug("Received API response as: \n")
                    debug_pp(data)
                    cls.cached_ip_results[ip] = data
                    return data

    @classmethod
    def estimate_gps_coordinates(cls) -> dict:
        """Get {latitude, longitude} by querying ipify's API
        """
        data: dict = cls.get_ip_result()
        if (not "lat" in data) or (not "lon" in data):
            if "error" in data:
                return data
            else:
                return {"error": "API call did not work as expected."}
        else:
            return {
                "latitude": data["lat"],
                "longitude": data["lon"]
            }

    @classmethod
    def get_nearest_city(cls):
        """Get city, region name by querying ipify's API
        """
        data: dict = cls.get_ip_result()
        if (not "city" in data) or (not "regionName" in data):
            if "error" in data:
                return data
            else:
                return {"error": "API call did not work as expected."}
        else:
            return { "city": f"{data['city']}, {data['regionName']}" }

    @classmethod
    def weather_foreast_from_gps_coorinates(cls, latitude: float, longitude: float) -> dict:
        """Get hour-by-hour weather forecast by querying openmeteo
        with today's date and the desired location in <lat,lon> coordinates
        """
        today = datetime.now().strftime("%Y-%m-%d")
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
           	"latitude": latitude,
           	"longitude": longitude,
            "timezone": "auto",
           	"hourly": ["temperature_2m", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "weather_code", "relative_humidity_2m", "precipitation_probability", "rain"],
           	"temperature_unit": "fahrenheit",
           	"precipitation_unit": "inch",
           	"start_date": today,
           	"end_date": today,
        }
        response = cls.openmeteocache_session.get(url, params=params)
        result = response.json()
        return result["hourly"]




toolbox = GeolocalInfoToolbox("geolocal_info", tools)
