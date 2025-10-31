"""
Tests for units conversion utilities.
"""

import pytest
from sdk.shared.units import UnitConverter, EnergyUnit, PowerUnit, TemperatureUnit


class TestEnergyConversion:
    """Test energy unit conversions."""

    def test_mwh_to_kwh(self):
        """Test MWh to kWh conversion."""
        result = UnitConverter.convert_energy(1.0, EnergyUnit.MWH, EnergyUnit.KWH)
        assert result == 1000.0

    def test_mwh_to_mmbtu(self):
        """Test MWh to MMBtu conversion."""
        result = UnitConverter.convert_energy(1.0, EnergyUnit.MWH, EnergyUnit.MMBTU)
        assert pytest.approx(result, 0.001) == 3.412142

    def test_kwh_to_mwh(self):
        """Test kWh to MWh conversion."""
        result = UnitConverter.convert_energy(1000.0, EnergyUnit.KWH, EnergyUnit.MWH)
        assert result == 1.0

    def test_same_unit(self):
        """Test conversion to same unit."""
        result = UnitConverter.convert_energy(100.0, EnergyUnit.MWH, EnergyUnit.MWH)
        assert result == 100.0

    def test_string_units(self):
        """Test conversion with string units."""
        result = UnitConverter.convert_energy(1.0, "MWh", "kWh")
        assert result == 1000.0


class TestPowerConversion:
    """Test power unit conversions."""

    def test_mw_to_kw(self):
        """Test MW to kW conversion."""
        result = UnitConverter.convert_power(1.0, PowerUnit.MW, PowerUnit.KW)
        assert result == 1000.0

    def test_mw_to_gw(self):
        """Test MW to GW conversion."""
        result = UnitConverter.convert_power(1000.0, PowerUnit.MW, PowerUnit.GW)
        assert result == 1.0

    def test_kw_to_mw(self):
        """Test kW to MW conversion."""
        result = UnitConverter.convert_power(1000.0, PowerUnit.KW, PowerUnit.MW)
        assert result == 1.0


class TestTemperatureConversion:
    """Test temperature unit conversions."""

    def test_fahrenheit_to_celsius(self):
        """Test F to C conversion."""
        result = UnitConverter.convert_temperature(32.0, TemperatureUnit.FAHRENHEIT, TemperatureUnit.CELSIUS)
        assert pytest.approx(result, 0.001) == 0.0

    def test_celsius_to_fahrenheit(self):
        """Test C to F conversion."""
        result = UnitConverter.convert_temperature(0.0, TemperatureUnit.CELSIUS, TemperatureUnit.FAHRENHEIT)
        assert pytest.approx(result, 0.001) == 32.0

    def test_celsius_to_kelvin(self):
        """Test C to K conversion."""
        result = UnitConverter.convert_temperature(0.0, TemperatureUnit.CELSIUS, TemperatureUnit.KELVIN)
        assert pytest.approx(result, 0.001) == 273.15

    def test_fahrenheit_to_kelvin(self):
        """Test F to K conversion."""
        result = UnitConverter.convert_temperature(32.0, TemperatureUnit.FAHRENHEIT, TemperatureUnit.KELVIN)
        assert pytest.approx(result, 0.001) == 273.15


class TestISOPriceStandardization:
    """Test ISO price standardization."""

    def test_caiso_price(self):
        """Test CAISO price standardization."""
        result = UnitConverter.standardize_iso_price(45.5, "CAISO")
        assert result == 45.5

    def test_miso_price(self):
        """Test MISO price standardization."""
        result = UnitConverter.standardize_iso_price(35.2, "MISO")
        assert result == 35.2

    def test_cents_per_kwh(self):
        """Test conversion from cents/kWh."""
        result = UnitConverter.standardize_iso_price(4.5, "CAISO", unit="CENTS/KWH")
        assert result == 45.0

    def test_dollars_per_kwh(self):
        """Test conversion from $/kWh."""
        result = UnitConverter.standardize_iso_price(0.045, "CAISO", unit="$/KWH")
        assert result == 45.0
