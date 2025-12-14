import requests
from datetime import datetime
#import openmeteo_requests
import requests_cache

from flat_mcp_client.tools import Toolbox
from . import implements_tool
from flat_mcp_client import debug, debug_pp



# CUSTOM TOOL DEFINITIONS
specific_tool_definitions: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "estimate_gps_coordinates",
            "description": "Approximates GPS coordinates of our current location",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_city",
            "description": "Approximate the current town/city corresponding to our location",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_todays_weather_forecast",
            "description": (
                "Get today's hourly weather forecast by querying openmeteo with the"
                " desired location in <lat,lon> coordinates"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "Current latitude"},
                    "longitude": {"type": "number", "description": "Current longitude"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_future_weather_forecast",
            "description": (
                "Get the future hour-by-hour weather forecast by querying openmeteo"
                " with the date in YY-MM-DD format and the location in <lat,lon> coordinates"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "Current latitude"},
                    "longitude": {"type": "number", "description": "Current longitude"},
                    "date": {"type": "string", "description": "forecasted date"},
                },
                "required": ["latitude", "longitude", "date"],
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
    cached_ip_results: dict[str, dict[str, object]] = {}

    # class variable for the openmeteo session
    openmeteocache_session: requests_cache.CachedSession = requests_cache.CachedSession('.cache', expire_after = 3600)

    # and two helper methods
    @staticmethod
    def get_public_IP_address() -> str | None:
        """Get public IP address through free-for-noncomercial use API
        or return None if the API call fails
        """
        url = 'https://api.ipify.org?format=json'
        response = requests.get(url)
        if response.status_code == 200:
            data: dict[str, object] = response.json()  # type: ignore[assignment]  # pyright: ignore[reportAny]
            ip_obj = data.get("ip")
            return ip_obj if isinstance(ip_obj, str) else None
        return None

    @classmethod
    def get_ip_result(cls) -> dict[str, object]:
        """Call ipify's API to get location data from
        public IP address, using cached results when possible
        """
        ip = cls.get_public_IP_address()
        if not ip:
            return {"error": "connection error"}
        else:
            # use cached result if available
            ip_data: dict[str, object] = {}
            if ip in cls.cached_ip_results:
                ip_data = cls.cached_ip_results[ip]
                debug("Retrieved cached result: \n")
                debug_pp(ip_data)
                return ip_data
            else:
                url = f'http://ip-api.com/json/{ip}'
                response = requests.get(url)
                if response.status_code != 200:
                    return {"error": f"Status code: {response.status_code}"}
                else:
                    ip_data = response.json()  # type: ignore[assignment]  # pyright: ignore[reportAny]
                    debug("Received API response as: \n")
                    debug_pp(ip_data)
                    cls.cached_ip_results[ip] = ip_data
                    return ip_data

    @classmethod
    @implements_tool
    def estimate_gps_coordinates(cls) -> dict[str, object]:
        """Get {latitude, longitude} by querying ipify's API
        """
        data: dict[str, object] = cls.get_ip_result()
        if ("lat" not in data) or ("lon" not in data):
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
    @implements_tool
    def get_current_city(cls) -> dict[str, object]:
        """Get city, region name by querying ipify's API
        """
        data: dict[str, object] = cls.get_ip_result()
        if ("city" not in data) or ("regionName" not in data):
            if "error" in data:
                return data
            else:
                return {"error": "API call did not work as expected."}
        else:
            return { "city": f"{data['city']}, {data['regionName']}" }

    @classmethod
    @implements_tool
    def get_todays_weather_forecast(cls, latitude: float, longitude: float) -> dict[str, object]:
        """Get today's hour-by-hour weather forecast by querying openmeteo
        with the desired location in <lat,lon> coordinates
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
        response = cls.openmeteocache_session.get(url, params=params)  # pyright: ignore[reportUnknownMemberType]
        result: dict[str, object] = response.json()  # type: ignore[assignment]  # pyright: ignore[reportAny]
        hourly = result.get("hourly")
        if isinstance(hourly, dict):
            hourly_dict: dict[str, object] = hourly  # pyright: ignore[reportUnknownVariableType]
            return hourly_dict
        return {"error": "Unexpected response payload"}

    @classmethod
    @implements_tool
    def get_future_weather_forecast(cls, latitude: float, longitude: float, date: str) -> dict[str, object]:
        """Get the future hour-by-hour weather forecast by querying openmeteo
        with the date in YY-MM-DD format and the location in <lat,lon> coordinates

        Args:
            latitude: the latitude of the location
            longitude: the longitude of the location
            date: the date to forecast, in ISO format, e.g., 2030-04-15
        """
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
           	"latitude": latitude,
           	"longitude": longitude,
            "timezone": "auto",
           	"hourly": ["temperature_2m", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "weather_code", "relative_humidity_2m", "precipitation_probability", "rain"],
           	"temperature_unit": "fahrenheit",
           	"precipitation_unit": "inch",
           	"start_date": date,
           	"end_date": date,
        }
        response = cls.openmeteocache_session.get(url, params=params)  # pyright: ignore[reportUnknownMemberType]
        result: dict[str, object] = response.json()  # type: ignore[assignment]  # pyright: ignore[reportAny]
        hourly = result.get("hourly")
        if isinstance(hourly, dict):
            hourly_dict: dict[str, object] = hourly  # pyright: ignore[reportUnknownVariableType]
            return hourly_dict
        return {"error": "Unexpected response payload"}
