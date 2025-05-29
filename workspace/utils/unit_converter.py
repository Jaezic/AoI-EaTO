# utils/unit_converter.py
import math

class UnitConverter:
    """
    Unit conversion utility class providing static methods for various unit conversions.
    """
    @staticmethod
    def dbm_to_watts(dbm_value: float) -> float:
        """Convert dBm to Watts."""
        return 10**((dbm_value - 30) / 10)

    @staticmethod
    def watts_to_dbm(watts_value: float) -> float:
        """Convert Watts to dBm."""
        if watts_value <= 0:
            return -float('inf') # log is undefined
        return 10 * math.log10(watts_value * 1000)

    @staticmethod
    def db_to_linear(db_value: float) -> float:
        """Convert dB to linear value."""
        return 10**(db_value / 10)

    @staticmethod
    def linear_to_db(linear_value: float) -> float:
        """Convert linear value to dB."""
        if linear_value <= 0:
            return -float('inf')
        return 10 * math.log10(linear_value)

    @staticmethod
    def mhz_to_hz(mhz_value: float) -> float:
        """Convert MHz to Hz."""
        return mhz_value * 1e6

    @staticmethod
    def mbits_to_bits(mbits_value: float) -> float:
        """Convert Mbits to bits."""
        return mbits_value * 1e6

    # Add more unit conversion functions as needed