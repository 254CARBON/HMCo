"""
Central unit conversion utilities.

Provides standardized conversions for energy, power, temperature, and other units
used across ISO feeds, weather data, and market data.
"""

from enum import Enum
from typing import Union, Optional
import math


class EnergyUnit(Enum):
    """Energy units."""
    MWH = "MWh"
    KWH = "kWh"
    GWH = "GWh"
    MMBTU = "MMBtu"
    THERM = "therm"
    BTU = "Btu"
    JOULE = "J"
    MJ = "MJ"
    GJ = "GJ"


class PowerUnit(Enum):
    """Power units."""
    MW = "MW"
    KW = "kW"
    GW = "GW"
    W = "W"


class TemperatureUnit(Enum):
    """Temperature units."""
    FAHRENHEIT = "F"
    CELSIUS = "C"
    KELVIN = "K"


class VolumeUnit(Enum):
    """Volume units for gas."""
    M3 = "m3"
    MCF = "Mcf"
    BCF = "Bcf"
    TCF = "Tcf"


class UnitConverter:
    """Central unit converter for all feed data."""

    # Conversion factors
    MWH_TO_MMBTU = 3.412142  # 1 MWh = 3.412142 MMBtu
    MWH_TO_KWH = 1000.0
    MWH_TO_GWH = 0.001
    MW_TO_KW = 1000.0
    MW_TO_GW = 0.001
    
    # Gas conversions (approximate, varies by composition)
    MCF_TO_MMBTU = 1.037  # Standard approximation
    BCF_TO_MCF = 1000.0
    
    @staticmethod
    def convert_energy(
        value: float,
        from_unit: Union[EnergyUnit, str],
        to_unit: Union[EnergyUnit, str]
    ) -> float:
        """
        Convert energy between units.
        
        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit
            
        Returns:
            Converted value
        """
        if isinstance(from_unit, str):
            from_unit = EnergyUnit(from_unit)
        if isinstance(to_unit, str):
            to_unit = EnergyUnit(to_unit)
        
        if from_unit == to_unit:
            return value
        
        # Convert to MWh as base unit
        mwh_value = value
        if from_unit == EnergyUnit.KWH:
            mwh_value = value / UnitConverter.MWH_TO_KWH
        elif from_unit == EnergyUnit.GWH:
            mwh_value = value / UnitConverter.MWH_TO_GWH
        elif from_unit == EnergyUnit.MMBTU:
            mwh_value = value / UnitConverter.MWH_TO_MMBTU
        
        # Convert from MWh to target
        if to_unit == EnergyUnit.MWH:
            return mwh_value
        elif to_unit == EnergyUnit.KWH:
            return mwh_value * UnitConverter.MWH_TO_KWH
        elif to_unit == EnergyUnit.GWH:
            return mwh_value * UnitConverter.MWH_TO_GWH
        elif to_unit == EnergyUnit.MMBTU:
            return mwh_value * UnitConverter.MWH_TO_MMBTU
        
        raise ValueError(f"Unsupported conversion: {from_unit} to {to_unit}")

    @staticmethod
    def convert_power(
        value: float,
        from_unit: Union[PowerUnit, str],
        to_unit: Union[PowerUnit, str]
    ) -> float:
        """
        Convert power between units.
        
        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit
            
        Returns:
            Converted value
        """
        if isinstance(from_unit, str):
            from_unit = PowerUnit(from_unit)
        if isinstance(to_unit, str):
            to_unit = PowerUnit(to_unit)
        
        if from_unit == to_unit:
            return value
        
        # Convert to MW as base unit
        mw_value = value
        if from_unit == PowerUnit.KW:
            mw_value = value / UnitConverter.MW_TO_KW
        elif from_unit == PowerUnit.GW:
            mw_value = value / UnitConverter.MW_TO_GW
        elif from_unit == PowerUnit.W:
            mw_value = value / (UnitConverter.MW_TO_KW * 1000)
        
        # Convert from MW to target
        if to_unit == PowerUnit.MW:
            return mw_value
        elif to_unit == PowerUnit.KW:
            return mw_value * UnitConverter.MW_TO_KW
        elif to_unit == PowerUnit.GW:
            return mw_value * UnitConverter.MW_TO_GW
        elif to_unit == PowerUnit.W:
            return mw_value * UnitConverter.MW_TO_KW * 1000
        
        raise ValueError(f"Unsupported conversion: {from_unit} to {to_unit}")

    @staticmethod
    def convert_temperature(
        value: float,
        from_unit: Union[TemperatureUnit, str],
        to_unit: Union[TemperatureUnit, str]
    ) -> float:
        """
        Convert temperature between units.
        
        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit
            
        Returns:
            Converted value
        """
        if isinstance(from_unit, str):
            from_unit = TemperatureUnit(from_unit)
        if isinstance(to_unit, str):
            to_unit = TemperatureUnit(to_unit)
        
        if from_unit == to_unit:
            return value
        
        # Convert to Celsius as intermediate
        celsius_value = value
        if from_unit == TemperatureUnit.FAHRENHEIT:
            celsius_value = (value - 32) * 5 / 9
        elif from_unit == TemperatureUnit.KELVIN:
            celsius_value = value - 273.15
        
        # Convert from Celsius to target
        if to_unit == TemperatureUnit.CELSIUS:
            return celsius_value
        elif to_unit == TemperatureUnit.FAHRENHEIT:
            return celsius_value * 9 / 5 + 32
        elif to_unit == TemperatureUnit.KELVIN:
            return celsius_value + 273.15
        
        raise ValueError(f"Unsupported conversion: {from_unit} to {to_unit}")

    @staticmethod
    def convert_volume_to_energy(
        value: float,
        from_unit: Union[VolumeUnit, str],
        heat_content_mmbtu_per_mcf: float = 1.037
    ) -> float:
        """
        Convert gas volume to energy (MMBtu).
        
        Args:
            value: Volume value
            from_unit: Volume unit
            heat_content_mmbtu_per_mcf: Heat content conversion factor
            
        Returns:
            Energy in MMBtu
        """
        if isinstance(from_unit, str):
            from_unit = VolumeUnit(from_unit)
        
        # Convert to Mcf
        mcf_value = value
        if from_unit == VolumeUnit.BCF:
            mcf_value = value * UnitConverter.BCF_TO_MCF
        elif from_unit == VolumeUnit.TCF:
            mcf_value = value * UnitConverter.BCF_TO_MCF * 1000
        
        # Convert to MMBtu
        return mcf_value * heat_content_mmbtu_per_mcf

    @staticmethod
    def standardize_iso_price(
        value: float,
        iso: str,
        unit: Optional[str] = None
    ) -> float:
        """
        Standardize ISO price to $/MWh.
        
        Some ISOs report in different units or scales.
        
        Args:
            value: Price value
            iso: ISO name (CAISO, MISO, SPP, etc.)
            unit: Optional unit override
            
        Returns:
            Standardized price in $/MWh
        """
        iso_upper = iso.upper()
        
        # Most ISOs report in $/MWh already
        # Add special cases as needed
        
        if unit:
            if unit.upper() == "$/KWH":
                return value * 1000  # Convert $/kWh to $/MWh
            elif unit.upper() == "CENTS/KWH":
                return value * 10  # Convert cents/kWh to $/MWh
        
        return value
