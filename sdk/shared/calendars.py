"""
Calendar and timezone utilities for ISO markets and data feeds.

Handles DST transitions, ISO-specific holidays, and trading calendars.
"""

from datetime import datetime, date, timedelta, timezone
from typing import List, Optional, Set
import pytz
from enum import Enum


class ISOTimezone(Enum):
    """ISO market timezones."""
    CAISO = "America/Los_Angeles"
    MISO = "America/Chicago"
    SPP = "America/Chicago"
    ERCOT = "America/Chicago"
    PJM = "America/New_York"
    NYISO = "America/New_York"
    ISONE = "America/New_York"


class CalendarUtils:
    """Calendar utilities for ISO markets and feeds."""

    # US Federal Holidays (affect market operations)
    FIXED_HOLIDAYS = {
        (1, 1): "New Year's Day",
        (7, 4): "Independence Day",
        (11, 11): "Veterans Day",
        (12, 25): "Christmas Day",
    }

    @staticmethod
    def get_iso_timezone(iso: str) -> pytz.timezone:
        """
        Get timezone for ISO.
        
        Args:
            iso: ISO name
            
        Returns:
            Timezone object
        """
        iso_upper = iso.upper()
        if iso_upper in ISOTimezone.__members__:
            tz_name = ISOTimezone[iso_upper].value
            return pytz.timezone(tz_name)
        
        # Default to UTC for unknown ISOs
        return pytz.UTC

    @staticmethod
    def convert_to_iso_time(dt: datetime, iso: str) -> datetime:
        """
        Convert datetime to ISO local time.
        
        Args:
            dt: Datetime to convert
            iso: ISO name
            
        Returns:
            Datetime in ISO local time
        """
        tz = CalendarUtils.get_iso_timezone(iso)
        
        if dt.tzinfo is None:
            # Assume UTC if no timezone
            dt = pytz.UTC.localize(dt)
        
        return dt.astimezone(tz)

    @staticmethod
    def is_dst(dt: datetime, iso: str) -> bool:
        """
        Check if datetime falls in DST for ISO.
        
        Args:
            dt: Datetime to check
            iso: ISO name
            
        Returns:
            True if DST is active
        """
        tz = CalendarUtils.get_iso_timezone(iso)
        local_dt = CalendarUtils.convert_to_iso_time(dt, iso)
        return local_dt.dst() != timedelta(0)

    @staticmethod
    def get_dst_transitions(year: int, iso: str) -> List[datetime]:
        """
        Get DST transition dates for year.
        
        Args:
            year: Year
            iso: ISO name
            
        Returns:
            List of transition datetimes
        """
        tz = CalendarUtils.get_iso_timezone(iso)
        
        transitions = []
        
        # Check each day in the year for DST change
        current_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        prev_dst = None
        
        while current_date <= end_date:
            dt = datetime.combine(current_date, datetime.min.time())
            dt = tz.localize(dt)
            current_dst = dt.dst() != timedelta(0)
            
            if prev_dst is not None and current_dst != prev_dst:
                transitions.append(dt)
            
            prev_dst = current_dst
            current_date += timedelta(days=1)
        
        return transitions

    @staticmethod
    def is_nerc_holiday(dt: date) -> bool:
        """
        Check if date is a NERC (North American Electric Reliability Corporation) holiday.
        
        Args:
            dt: Date to check
            
        Returns:
            True if NERC holiday
        """
        # Fixed holidays
        if (dt.month, dt.day) in CalendarUtils.FIXED_HOLIDAYS:
            return True
        
        # Memorial Day (last Monday in May)
        if dt.month == 5 and dt.weekday() == 0:
            if dt.day > 24:  # Last week of May
                return True
        
        # Labor Day (first Monday in September)
        if dt.month == 9 and dt.weekday() == 0 and dt.day <= 7:
            return True
        
        # Thanksgiving (4th Thursday in November)
        if dt.month == 11 and dt.weekday() == 3:
            # Count Thursdays
            first_day = date(dt.year, 11, 1)
            thursday_count = 0
            current = first_day
            while current <= dt:
                if current.weekday() == 3:
                    thursday_count += 1
                current += timedelta(days=1)
            if thursday_count == 4:
                return True
        
        return False

    @staticmethod
    def get_trading_hours(iso: str) -> dict:
        """
        Get trading hours for ISO.
        
        Args:
            iso: ISO name
            
        Returns:
            Dict with trading hour info
        """
        iso_upper = iso.upper()
        
        # Most ISOs use hour-ending convention
        base = {
            "convention": "hour_ending",
            "hours_per_day": 24,
            "dst_spring_hours": 23,  # Spring forward: 23 hours
            "dst_fall_hours": 25,    # Fall back: 25 hours
        }
        
        if iso_upper == "CAISO":
            base["market_close"] = "13:00"  # 1 PM local time
        elif iso_upper in ["MISO", "SPP"]:
            base["market_close"] = "11:00"  # 11 AM local time
        elif iso_upper in ["PJM", "NYISO"]:
            base["market_close"] = "10:30"  # 10:30 AM local time
        
        return base

    @staticmethod
    def hour_ending_to_hour_beginning(he: int) -> int:
        """
        Convert hour-ending to hour-beginning.
        
        Args:
            he: Hour-ending (1-24 or 1-25)
            
        Returns:
            Hour-beginning (0-23 or 0-24)
        """
        return he - 1

    @staticmethod
    def hour_beginning_to_hour_ending(hb: int) -> int:
        """
        Convert hour-beginning to hour-ending.
        
        Args:
            hb: Hour-beginning (0-23)
            
        Returns:
            Hour-ending (1-24)
        """
        return hb + 1

    @staticmethod
    def get_business_days(
        start_date: date,
        end_date: date,
        exclude_nerc_holidays: bool = True
    ) -> List[date]:
        """
        Get list of business days between dates.
        
        Args:
            start_date: Start date
            end_date: End date
            exclude_nerc_holidays: Exclude NERC holidays
            
        Returns:
            List of business day dates
        """
        business_days = []
        current = start_date
        
        while current <= end_date:
            # Skip weekends
            if current.weekday() < 5:
                # Check holidays
                if exclude_nerc_holidays and CalendarUtils.is_nerc_holiday(current):
                    pass
                else:
                    business_days.append(current)
            
            current += timedelta(days=1)
        
        return business_days

    @staticmethod
    def align_event_time(
        dt: datetime,
        iso: str,
        round_to_minutes: int = 5
    ) -> datetime:
        """
        Align event time to standard intervals.
        
        Handles clock skew and resampling for ISO data.
        
        Args:
            dt: Datetime to align
            iso: ISO name
            round_to_minutes: Round to nearest N minutes
            
        Returns:
            Aligned datetime
        """
        # Convert to ISO local time
        local_dt = CalendarUtils.convert_to_iso_time(dt, iso)
        
        # Round to nearest interval
        minutes = (local_dt.minute // round_to_minutes) * round_to_minutes
        aligned = local_dt.replace(minute=minutes, second=0, microsecond=0)
        
        return aligned

    @staticmethod
    def get_settlement_periods(dt: date, iso: str) -> int:
        """
        Get number of settlement periods for date.
        
        Handles DST transitions.
        
        Args:
            dt: Date
            iso: ISO name
            
        Returns:
            Number of settlement periods (usually 24, 23, or 25)
        """
        trading_hours = CalendarUtils.get_trading_hours(iso)
        
        # Check if this is a DST transition day
        year = dt.year
        transitions = CalendarUtils.get_dst_transitions(year, iso)
        
        for transition in transitions:
            if transition.date() == dt:
                # Spring forward: 23 hours
                if CalendarUtils.is_dst(transition + timedelta(hours=1), iso):
                    return trading_hours["dst_spring_hours"]
                # Fall back: 25 hours
                else:
                    return trading_hours["dst_fall_hours"]
        
        return trading_hours["hours_per_day"]
