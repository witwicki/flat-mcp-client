from datetime import datetime, date, timedelta
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import humanize

from flat_mcp_client.tools import Toolbox
from . import implements_tool
from flat_mcp_client import debug, debug_pp


class TimeAndDateToolbox(Toolbox):
    """Simplest example of tools."""

    @staticmethod
    @implements_tool
    def day_of_the_week() -> str:
        """Get the day of the week today"""
        return datetime.now().strftime('%A')

    @staticmethod
    @implements_tool
    def date_today() -> str:
        """Get today's date"""
        return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    @implements_tool
    def clock(timezone: Optional[str] = None) -> str:
        """Get the current time here, or optionally, in a specified timezone or location.  Call this function without any arguments
        to the the local time.

        Args:
            timezone: a string indicating the timezone code or the IANA time zone or the
        """
        if not timezone:
            return datetime.now().strftime("%I:%M%p")
        valid_timezone = ZoneInfo("America/Los_Angeles")
        try:
            # try to get the timzone, first by the code
            valid_timezone = ZoneInfo(timezone)
        except ZoneInfoNotFoundError:
            # and next by the location (treating the `timezone` as a location)
            try:
                geolocator = Nominatim(user_agent="time_and_date_agent")
                location = geolocator.geocode(timezone)
                tz_name = TimezoneFinder().timezone_at(lng=location.longitude, lat=location.latitude) # type: ignore missing fields
                valid_timezone = ZoneInfo(tz_name) # type: ignore possibility that tz_name is None
            except Exception as e:
                debug(f"Error looking determining timezone: {e}")
                return "Unknown.  Please specify a timezone code for that location."
        # return the time in the timeone
        return datetime.now(valid_timezone).strftime("%I:%M%p")

    @staticmethod
    @implements_tool
    def time_remaining_until(time: str, date: str = date.today().isoformat(), precisely: bool = False) -> str:
        """Get the amount of time remaining until a specified time.  Optionally, you can also specify a date.

        Args:
            time: the deadline time, in ISO format, e.g., 17:00:00
            date: the deadline date, in ISO format, e.g., 2030-04-15
            precisely: wheher or not to answer to the highest degree of precision
        """
        try:
            deadline = datetime.fromisoformat(f"{date} {time}")
            delta = deadline - datetime.now()
            # did the time already pass?
            if delta < timedelta():
                return "No time remains.  The queried time and date happens to be in the past!"
            # answer with how much time is left
            if precisely:
                return humanize.precisedelta(delta)
            else:
                return humanize.naturaldelta(delta)
        except:
            return "Error: please check the ISO formatting of the arguments"
