#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modbus Protocol Specification

This module contains Modbus protocol constants, function codes, exception codes,
and protocol-specific parsing functions.

Description:
    Complete Modbus protocol specification including TCP and RTU variants,
    function codes, exception codes, and data extraction functions.
"""

from typing import Dict, Optional, Any
import struct

# Modbus Function Codes
MODBUS_FUNCTION_CODES = {
    0x01: "Read Coils",
    0x02: "Read Discrete Inputs",
    0x03: "Read Holding Registers",
    0x04: "Read Input Registers",
    0x05: "Write Single Coil",
    0x06: "Write Single Register",
    0x07: "Read Exception Status",
    0x08: "Diagnostics",
    0x0B: "Get Comm Event Counter",
    0x0C: "Get Comm Event Log",
    0x0F: "Write Multiple Coils",
    0x10: "Write Multiple Registers",
    0x11: "Report Server ID",
    0x14: "Read File Record",
    0x15: "Write File Record",
    0x16: "Mask Write Register",
    0x17: "Read/Write Multiple Registers",
    0x18: "Read FIFO Queue",
    0x2B: "Encapsulated Interface Transport"
}

# Modbus Exception Codes
MODBUS_EXCEPTION_CODES = {
    0x01: "Illegal Function",
    0x02: "Illegal Data Address",
    0x03: "Illegal Data Value",
    0x04: "Server Device Failure",
    0x05: "Acknowledge",
    0x06: "Server Device Busy",
    0x08: "Memory Parity Error",
    0x0A: "Gateway Path Unavailable",
    0x0B: "Gateway Target Device Failed"
}

# Modbus Constants
MODBUS_TCP_HEADER_SIZE = 7  # MBAP Header (6 bytes) + Unit ID (1 byte)
MODBUS_RTU_HEADER_SIZE = 1  # Unit ID only
MODBUS_RTU_CRC_SIZE = 2     # CRC-16

# Data Function Codes (functions that carry actual data)
MODBUS_DATA_FUNCTION_CODES = {
    0x01, 0x02, 0x03, 0x04,  # Read functions
    0x05, 0x06,              # Write single functions
    0x0F, 0x10,              # Write multiple functions
    0x14, 0x15,              # File record functions
    0x16, 0x17               # Advanced functions
}

# Protocol Numbers
MODBUS_TCP_DEFAULT_PORT = 502
MODBUS_TCP_PROTOCOL_ID = 0


# Helper functions
def get_modbus_function_name(function_code: int) -> str:
    """Get function name from function code."""
    return MODBUS_FUNCTION_CODES.get(function_code, f"Unknown_Function_0x{function_code:02X}")


def get_modbus_exception_name(exception_code: int) -> str:
    """Get exception name from exception code."""
    return MODBUS_EXCEPTION_CODES.get(exception_code, f"Unknown_Exception_0x{exception_code:02X}")


def is_modbus_exception(function_code: int) -> bool:
    """Check if function code indicates an exception response."""
    return (function_code & 0x80) != 0


def get_original_function_code(exception_code: int) -> int:
    """Get the original function code from an exception response."""
    return exception_code & 0x7F


def is_data_function(function_code: int) -> bool:
    """Check if function code is a data-carrying function."""
    return function_code in MODBUS_DATA_FUNCTION_CODES


# Protocol parsing functions
def parse_modbus_tcp_header(tcp_payload: bytes) -> Optional[Dict[str, Any]]:
    """
    Parse Modbus TCP header (MBAP + Unit ID) from TCP payload.
    
    Args:
        tcp_payload: TCP payload bytes
        
    Returns:
        Dictionary with parsed Modbus header fields
    """
    try:
        if len(tcp_payload) < MODBUS_TCP_HEADER_SIZE:
            return None
        
        # Parse MBAP Header (6 bytes) + Unit ID (1 byte)
        transaction_id = int.from_bytes(tcp_payload[0:2], byteorder='big')
        protocol_id = int.from_bytes(tcp_payload[2:4], byteorder='big')
        length = int.from_bytes(tcp_payload[4:6], byteorder='big')
        unit_id = tcp_payload[6]
        
        # Parse PDU if available
        function_code = None
        function_name = None
        is_exception = False
        is_data_func = False
        
        if len(tcp_payload) > MODBUS_TCP_HEADER_SIZE:
            function_code = tcp_payload[MODBUS_TCP_HEADER_SIZE]
            is_exception = is_modbus_exception(function_code)
            if is_exception:
                original_func = get_original_function_code(function_code)
                function_name = f"{get_modbus_function_name(original_func)}_Exception"
            else:
                function_name = get_modbus_function_name(function_code)
                is_data_func = is_data_function(function_code)
        
        return {
            'protocol': 'Modbus',
            'transaction_id': transaction_id,
            'protocol_id': protocol_id,
            'length': length,
            'unit_id': unit_id,
            'function_code': function_code,
            'function_name': function_name,
            'is_exception': is_exception,
            'is_data_function': is_data_func,
            'header_size': MODBUS_TCP_HEADER_SIZE,
            'is_valid_modbus': protocol_id == MODBUS_TCP_PROTOCOL_ID
        }
        
    except Exception as e:
        return None


def extract_pure_data_from_modbus_tcp(tcp_payload: bytes, modbus_header_info: dict) -> Optional[bytes]:
    """
    Extract pure data from Modbus TCP payload.
    
    Args:
        tcp_payload: TCP payload bytes
        modbus_header_info: Modbus header information
    Returns:
        Pure data bytes or None if extraction fails
    """
    try:
        if len(tcp_payload) < MODBUS_TCP_HEADER_SIZE:
            return None
        
        # Parse header first
        if not modbus_header_info or not modbus_header_info['is_valid_modbus']:
            return None
        
        # Extract Modbus PDU (skip MBAP header + Unit ID)
        if len(tcp_payload) > MODBUS_TCP_HEADER_SIZE:
            modbus_pdu = tcp_payload[MODBUS_TCP_HEADER_SIZE:]
            
            # For data functions, return data part (skip function code)
            if len(modbus_pdu) > 1 and modbus_header_info['is_data_function']:
                return modbus_pdu[1:]  # Skip function code, return data
            elif len(modbus_pdu) > 0:
                return modbus_pdu      # Return entire PDU for non-data functions
        
        return None
        
    except Exception as e:
        return None


def parse_modbus_rtu_frame(rtu_frame: bytes) -> Optional[Dict[str, Any]]:
    """
    Parse Modbus RTU frame.
    
    Args:
        rtu_frame: Complete RTU frame bytes including CRC
        
    Returns:
        Dictionary with parsed RTU frame fields
    """
    try:
        if len(rtu_frame) < MODBUS_RTU_HEADER_SIZE + MODBUS_RTU_CRC_SIZE + 1:  # Min: Unit ID + Function + CRC
            return None
        
        # Extract components
        unit_id = rtu_frame[0]
        function_code = rtu_frame[1] if len(rtu_frame) > 1 else None
        
        # Extract CRC (last 2 bytes)
        crc_bytes = rtu_frame[-2:]
        crc = int.from_bytes(crc_bytes, byteorder='little')  # RTU uses little-endian CRC
        
        # Extract data (everything between function code and CRC)
        data_bytes = rtu_frame[2:-2] if len(rtu_frame) > 4 else b''
        
        # Parse function info
        function_name = None
        is_exception = False
        is_data_func = False
        
        if function_code is not None:
            is_exception = is_modbus_exception(function_code)
            if is_exception:
                original_func = get_original_function_code(function_code)
                function_name = f"{get_modbus_function_name(original_func)}_Exception"
            else:
                function_name = get_modbus_function_name(function_code)
                is_data_func = is_data_function(function_code)
        
        return {
            'unit_id': unit_id,
            'function_code': function_code,
            'function_name': function_name,
            'is_exception': is_exception,
            'is_data_function': is_data_func,
            'data': data_bytes,
            'crc': crc,
            'frame_length': len(rtu_frame)
        }
        
    except Exception as e:
        return None


# Example usage
if __name__ == "__main__":
    # Test Modbus-specific dictionaries and functions
    print("Modbus Function Code Examples:")
    test_functions = [0x01, 0x03, 0x06, 0x10, 0x16, 0x17]
    for func in test_functions:
        print(f"  0x{func:02X}: {get_modbus_function_name(func)}")
        print(f"    Is Data Function: {is_data_function(func)}")
    
    print("\nModbus Exception Code Examples:")
    test_exceptions = [0x01, 0x02, 0x03, 0x04, 0x06]
    for exc in test_exceptions:
        print(f"  0x{exc:02X}: {get_modbus_exception_name(exc)}")
    
    print("\nModbus Exception Response Examples:")
    test_exception_responses = [0x81, 0x83, 0x86, 0x90]  # Function codes with exception bit set
    for exc_resp in test_exception_responses:
        original = get_original_function_code(exc_resp)
        print(f"  0x{exc_resp:02X}: Exception response for {get_modbus_function_name(original)} (0x{original:02X})")
    
    print(f"\nModbus Protocol Constants:")
    print(f"  TCP Header Size: {MODBUS_TCP_HEADER_SIZE} bytes")
    print(f"  RTU Header Size: {MODBUS_RTU_HEADER_SIZE} bytes")
    print(f"  RTU CRC Size: {MODBUS_RTU_CRC_SIZE} bytes")
    print(f"  Default TCP Port: {MODBUS_TCP_DEFAULT_PORT}")
    print(f"  TCP Protocol ID: {MODBUS_TCP_PROTOCOL_ID}")
    
    print(f"\nData Function Codes: {MODBUS_DATA_FUNCTION_CODES}")
    
    # Test Modbus TCP header parsing
    print(f"\nTesting Modbus TCP Header Parsing:")
    sample_modbus_header = b'\x00\x01\x00\x00\x00\x06\x01\x03\x00\x00\x00\x02'
    parsed = parse_modbus_tcp_header(sample_modbus_header)
    if parsed:
        print(f"  Transaction ID: {parsed['transaction_id']}")
        print(f"  Protocol ID: {parsed['protocol_id']}")
        print(f"  Length: {parsed['length']}")
        print(f"  Unit ID: {parsed['unit_id']}")
        print(f"  Function: {parsed['function_name']} (0x{parsed['function_code']:02X})")
        print(f"  Is Data Function: {parsed['is_data_function']}")
        print(f"  Is Valid Modbus: {parsed['is_valid_modbus']}")
    else:
        print("  Failed to parse header")