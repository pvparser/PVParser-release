"""
Protocol Factory - Unified Protocol Management

This module provides a factory pattern for managing different industrial
protocols. It allows dynamic selection of protocol handlers based on 
protocol type configuration.

Author: PVParser Project
Created: 2025-08-17
Last Modified: 2025-08-17
Version: 1.0.0
"""

from typing import Dict, List, Optional, Callable, Any
from scapy.layers.inet import IP, TCP
from scapy.layers.l2 import Ether
import os
import sys

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basis.ics_basis import ENIP, MODBUS, MMS
from basis.specification_enip import ENIP_HEADER_SIZE, ENIP_DATA_COMMANDS, get_enip_command_name, get_enip_status_description, parse_enip_header, extract_enip_payload
from basis.specification_cip import CIP_MIN_HEADER_SIZE, CIP_DATA_SERVICES, get_cip_service_name, parse_cip_header, extract_pure_data_from_cip
from basis.specification_modbus import MODBUS_TCP_HEADER_SIZE, MODBUS_DATA_FUNCTION_CODES, get_modbus_function_name, get_modbus_exception_name, parse_modbus_tcp_header, extract_pure_data_from_modbus_tcp
from basis.specification_mms import MMS_DATA_COMMANDS, parse_mms_header, extract_pure_data_from_mms

SUPPORTED_PROTOCOLS = [ENIP, MODBUS, MMS]

# Protocol Configuration
PROTOCOL_CONFIG = {
    ENIP: {
        "name": "EtherNet/IP",
        "extractor": extract_pure_data_from_cip,
        "parser": parse_enip_header,
        "sub_parser": parse_cip_header,
        "header_size": ENIP_HEADER_SIZE,
        "min_sub_header_size": CIP_MIN_HEADER_SIZE,
        "data_commands": ENIP_DATA_COMMANDS,
        "data_services": CIP_DATA_SERVICES,
        "command_name_func": get_enip_command_name,
        "service_name_func": get_cip_service_name,
        "status_name_func": get_enip_status_description
    },
    MODBUS: {
        "name": "Modbus/TCP",
        "extractor": extract_pure_data_from_modbus_tcp,
        "parser": parse_modbus_tcp_header,
        "sub_parser": None,
        "header_size": MODBUS_TCP_HEADER_SIZE,
        "min_sub_header_size": 0,
        "data_commands": MODBUS_DATA_FUNCTION_CODES,
        "data_services": set(),
        "command_name_func": get_modbus_function_name,
        "service_name_func": None,
        "status_name_func": get_modbus_exception_name
    },
    MMS: {
        "name": "MMS",
        "extractor": extract_pure_data_from_mms,
        "parser": parse_mms_header,
        "sub_parser": None,
        "header_size": None,
        "min_sub_header_size": 0,
        "data_commands": MMS_DATA_COMMANDS,
        "data_services": set(),
        "command_name_func": None,
        "service_name_func": None,
        "status_name_func": None
    }
}


class ProtocolFactory:
    """
    Factory class for managing different industrial protocols.
    
    This class provides a unified interface for working with different
    protocol types and their specific handlers.
    """
    
    def __init__(self):
        """Initialize the protocol factory."""
        self.current_protocol = None
        self.current_config = None
    
    @staticmethod
    def get_supported_protocols() -> List[str]:
        """Get list of supported protocol types."""
        return list(SUPPORTED_PROTOCOLS)
    
    @staticmethod
    def is_protocol_supported(protocol_type: str) -> bool:
        """Check if a protocol type is supported."""
        return protocol_type in SUPPORTED_PROTOCOLS
    
    @staticmethod
    def get_protocol_config(protocol_type: str) -> Dict[str, Any]:
        """
        Get protocol configuration by type.
        
        Args:
            protocol_type: Protocol type string
            
        Returns:
            Protocol configuration dictionary
            
        Raises:
            ValueError: If protocol type is not supported
        """
        if protocol_type not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"Unsupported protocol type: {protocol_type}. "
                            f"Supported types: {', '.join(SUPPORTED_PROTOCOLS)}")
        
        return PROTOCOL_CONFIG[protocol_type].copy()
    
    @staticmethod
    def get_protocol_info() -> Dict[str, Dict[str, str]]:
        """
        Get information about all supported protocols.
        
        Returns:
            Dictionary with protocol information
        """
        return {
            protocol_type: {
                "name": config["name"],
                "description": config["description"]
            }
            for protocol_type, config in PROTOCOL_CONFIG.items()
        }
        
    def set_protocol(self, protocol_type: str) -> bool:
        """
        Set the current protocol type.
        
        Args:
            protocol_type: Protocol type string
            
        Returns:
            True if protocol was set successfully, False otherwise
        """
        if not self.is_protocol_supported(protocol_type):
            print(f"Unsupported protocol type: {protocol_type}")
            return False
        
        self.current_protocol = protocol_type
        self.current_config = PROTOCOL_CONFIG[protocol_type].copy()
        print(f"Protocol set to: {self.current_config['name']}")
        return True
    
    def get_current_protocol(self) -> Optional[str]:
        """Get the current protocol type."""
        return self.current_protocol
    
    def get_current_config(self) -> Optional[Dict[str, Any]]:
        """Get the current protocol configuration."""
        return self.current_config
    
    def parse_header(self, packet) -> Optional[Dict[str, Any]]:
        """
        Parse protocol header using the current protocol from a packet object.
        
        Args:
            packet: Scapy packet object
            
        Returns:
            Parsed header dictionary or None if parsing fails
        """
        if not self.current_config:
            print("No protocol set")
            return None
        
        parser = self.current_config.get('parser')
        if not parser:
            print(f"No parser available for {self.current_protocol}")
            return None
        
        # Extract TCP payload from packet first
        if not (IP in packet and TCP in packet):
            return None
        
        tcp_layer = packet[TCP]
        if hasattr(tcp_layer, 'payload') and tcp_layer.payload:
            tcp_payload = bytes(tcp_layer.payload)
            return parser(tcp_payload)
        
        return None
    
    def parse_sub_header(self, data: bytes, endianness: str) -> Optional[Dict[str, Any]]:
        """
        Parse sub-protocol header using the current protocol.
        
        Args:
            data: Data bytes
            
        Returns:
            Parsed sub-header dictionary or None if parsing fails
        """
        if not self.current_config:
            print("No protocol set")
            return None
        
        sub_parser = self.current_config.get('sub_parser')
        if not sub_parser:
            return None  # Not all protocols have sub-parsers
        
        return sub_parser(data, endianness)
    
    def extract_pure_data(self, packet) -> Optional[bytes]:
        """
        Extract pure data using the current protocol from a packet object.
        
        Args:
            packet: Scapy packet object
            
        Returns:
            Tuple containing protocol info and pure data bytes, or None if extraction fails
        """
        if not self.current_config:
            print("No protocol set")
            return None, None
        
        extractor = self.current_config.get('extractor')
        if not extractor:
            print(f"No extractor available for {self.current_protocol}")
            return None, None
        
        # Extract TCP payload from packet first
        if not (IP in packet and TCP in packet):
            return None
        
        tcp_layer = packet[TCP]
        if hasattr(tcp_layer, 'payload') and tcp_layer.payload:
            tcp_payload = bytes(tcp_layer.payload)
            
            # For ENIP protocol, we need to parse ENIP header first, then extract payload
            if self.current_protocol == ENIP:
                # Parse ENIP header
                enip_header_info = self.parse_header(packet)
                if not enip_header_info:
                    return None, None
                
                # Extract ENIP payload
                enip_payload = extract_enip_payload(tcp_payload, enip_header_info)
                if not enip_payload:
                    return None, None
                
                # Use CIP extractor to get pure data from ENIP payload
                if enip_header_info['is_data_command']:
                    cip_header_info = self.parse_sub_header(enip_payload, enip_header_info['endianness'])
                    cip_data = extractor(enip_payload, cip_header_info)  # Part of the CIP payload is CIP data, exclude service code etc.
                    return cip_header_info, cip_data
                else:
                    return enip_header_info, enip_payload
            elif self.current_protocol == MODBUS:  # TODO test
                modbus_header_info = self.parse_header(packet)
                if not modbus_header_info:
                    return None, None
                
                modbus_payload = extractor(tcp_payload, modbus_header_info)
                return modbus_header_info, modbus_payload
            elif self.current_protocol == MMS:
                mms_payload = extractor(packet)
                return None, mms_payload
        
        return None


# Convenience functions for backward compatibility
def get_protocol_extractor(protocol_type: str) -> Optional[Callable]:
    """Get the data extractor function for a protocol type."""
    try:
        config = ProtocolFactory.get_protocol_config(protocol_type)
        return config.get('extractor')
    except ValueError:
        return None


def get_protocol_parser(protocol_type: str) -> Optional[Callable]:
    """Get the header parser function for a protocol type."""
    try:
        config = ProtocolFactory.get_protocol_config(protocol_type)
        return config.get('parser')
    except ValueError:
        return None


def get_protocol_sub_parser(protocol_type: str) -> Optional[Callable]:
    """Get the sub-protocol header parser function for a protocol type."""
    try:
        config = ProtocolFactory.get_protocol_config(protocol_type)
        return config.get('sub_parser')
    except ValueError:
        return None


# Example usage
def main():
    """Example usage of the ProtocolFactory."""
    print("Protocol Factory Demo")
    print("=" * 50)
    
    # Create factory instance
    factory = ProtocolFactory()
    
    # Show supported protocols
    print("\nSupported Protocols:")
    for protocol_type, info in factory.get_protocol_info().items():
        print(f"  {protocol_type.upper()}: {info['name']} - {info['description']}")
    
    # Test ENIP protocol
    print(f"\nTesting ENIP Protocol:")
    if factory.set_protocol(PROTOCOL_TYPE_ENIP):
        config = factory.get_current_config()
        print(f"  Current Protocol: {config['name']}")
        print(f"  Header Size: {config['header_size']} bytes")
        print(f"  Data Commands: {len(config['data_commands'])}")
    
    # Test Modbus protocol
    print(f"\nTesting Modbus Protocol:")
    if factory.set_protocol(PROTOCOL_TYPE_MODBUS):
        config = factory.get_current_config()
        print(f"  Current Protocol: {config['name']}")
        print(f"  Header Size: {config['header_size']} bytes")
        print(f"  Data Function Codes: {config['data_commands']}")
    
    # Test unsupported protocol
    print(f"\nTesting Unsupported Protocol:")
    if not factory.set_protocol("unsupported"):
        print("  Failed to set unsupported protocol (expected)")
    
    print(f"\nDemo completed!")


if __name__ == "__main__":
    main()
