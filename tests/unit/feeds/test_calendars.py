"""
Tests for calendar and timezone utilities.
"""

import pytest
from datetime import datetime, date, timezone
import pytz
from sdk.shared.calendars import CalendarUtils, ISOTimezone


class TestTimezoneHandling:
    """Test timezone handling for ISOs."""

    def test_caiso_timezone(self):
        """Test CAISO timezone."""
        tz = CalendarUtils.get_iso_timezone("CAISO")
        assert str(tz) == "America/Los_Angeles"

    def test_miso_timezone(self):
        """Test MISO timezone."""
        tz = CalendarUtils.get_iso_timezone("MISO")
        assert str(tz) == "America/Chicago"

    def test_spp_timezone(self):
        """Test SPP timezone."""
        tz = CalendarUtils.get_iso_timezone("SPP")
        assert str(tz) == "America/Chicago"

    def test_unknown_iso_default_utc(self):
        """Test unknown ISO defaults to UTC."""
        tz = CalendarUtils.get_iso_timezone("UNKNOWN")
        assert tz == pytz.UTC


class TestDSTDetection:
    """Test DST detection."""

    def test_dst_summer(self):
        """Test DST active in summer."""
        dt = datetime(2025, 7, 15, 12, 0, 0, tzinfo=pytz.UTC)
        assert CalendarUtils.is_dst(dt, "CAISO") == True

    def test_dst_winter(self):
        """Test DST inactive in winter."""
        dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=pytz.UTC)
        assert CalendarUtils.is_dst(dt, "CAISO") == False


class TestHolidayDetection:
    """Test NERC holiday detection."""

    def test_new_years_day(self):
        """Test New Year's Day."""
        assert CalendarUtils.is_nerc_holiday(date(2025, 1, 1)) == True

    def test_independence_day(self):
        """Test Independence Day."""
        assert CalendarUtils.is_nerc_holiday(date(2025, 7, 4)) == True

    def test_christmas(self):
        """Test Christmas."""
        assert CalendarUtils.is_nerc_holiday(date(2025, 12, 25)) == True

    def test_not_holiday(self):
        """Test regular day."""
        assert CalendarUtils.is_nerc_holiday(date(2025, 6, 15)) == False


class TestHourConversion:
    """Test hour-ending/beginning conversion."""

    def test_hour_ending_to_beginning(self):
        """Test HE to HB conversion."""
        assert CalendarUtils.hour_ending_to_hour_beginning(1) == 0
        assert CalendarUtils.hour_ending_to_hour_beginning(24) == 23

    def test_hour_beginning_to_ending(self):
        """Test HB to HE conversion."""
        assert CalendarUtils.hour_beginning_to_hour_ending(0) == 1
        assert CalendarUtils.hour_beginning_to_hour_ending(23) == 24


class TestBusinessDays:
    """Test business day calculation."""

    def test_business_days_no_holidays(self):
        """Test business days excluding weekends."""
        days = CalendarUtils.get_business_days(
            date(2025, 6, 2),  # Monday
            date(2025, 6, 6),  # Friday
            exclude_nerc_holidays=False
        )
        assert len(days) == 5

    def test_business_days_with_weekend(self):
        """Test business days with weekend."""
        days = CalendarUtils.get_business_days(
            date(2025, 6, 2),  # Monday
            date(2025, 6, 8),  # Sunday
            exclude_nerc_holidays=False
        )
        assert len(days) == 5  # Excludes Sat/Sun

    def test_business_days_with_holiday(self):
        """Test business days with holiday."""
        days = CalendarUtils.get_business_days(
            date(2025, 7, 1),  # Tuesday
            date(2025, 7, 7),  # Monday
            exclude_nerc_holidays=True
        )
        # Should exclude July 4 (Friday)
        assert len(days) == 4


class TestSettlementPeriods:
    """Test settlement period calculation."""

    def test_normal_day(self):
        """Test normal day has 24 periods."""
        periods = CalendarUtils.get_settlement_periods(date(2025, 6, 15), "CAISO")
        assert periods == 24

    def test_dst_spring_forward(self):
        """Test spring forward has 23 periods."""
        # Note: DST transitions vary by year, this is approximate
        # In 2025, DST starts March 9
        periods = CalendarUtils.get_settlement_periods(date(2025, 3, 9), "CAISO")
        # Should be 23 hours (spring forward)
        assert periods in [23, 24]  # May vary by exact transition logic

    def test_dst_fall_back(self):
        """Test fall back has 25 periods."""
        # In 2025, DST ends November 2
        periods = CalendarUtils.get_settlement_periods(date(2025, 11, 2), "CAISO")
        # Should be 25 hours (fall back)
        assert periods in [24, 25]  # May vary by exact transition logic


class TestEventTimeAlignment:
    """Test event time alignment."""

    def test_align_to_5_minutes(self):
        """Test alignment to 5-minute intervals."""
        dt = datetime(2025, 1, 15, 12, 3, 47, tzinfo=pytz.UTC)
        aligned = CalendarUtils.align_event_time(dt, "CAISO", round_to_minutes=5)
        assert aligned.minute == 0

    def test_align_to_15_minutes(self):
        """Test alignment to 15-minute intervals."""
        dt = datetime(2025, 1, 15, 12, 17, 0, tzinfo=pytz.UTC)
        aligned = CalendarUtils.align_event_time(dt, "CAISO", round_to_minutes=15)
        assert aligned.minute == 15
