# EtherNet/IP (ENIP) Protocol Specification
# Based on CIP Specification Volume 2 and EtherNet/IP Specification

ENIP_COMMANDS = {
    # Basic Encapsulation Commands
    0x0004: "ListServices",                 # List Services
    0x0063: "ListIdentity",                 # List Identity
    0x0064: "ListInterfaces",               # List Interfaces
    0x0065: "RegisterSession",              # Register Session
    0x0066: "UnregisterSession",            # Unregister Session
    0x006F: "SendRRData",                   # Send Request/Reply Data
    0x0070: "SendUnitData",                 # Send Unit Data

    # Extended Commands
    0x0002: "ListTargets",                  # List Targets (rarely used)
    0x0003: "ListParameters",               # List Parameters (rarely used)
    0x0005: "ListCapabilities",             # List Capabilities (rarely used)
    0x0006: "StartDTLS",                    # Start DTLS Session
}

# ENIP Status Codes
ENIP_STATUS_CODES = {
    0x0000: "Success",
    0x0001: "Invalid Command",
    0x0002: "Insufficient Memory",
    0x0003: "Incorrect Data",
    0x0004: "Invalid Session Handle",
    0x0005: "Invalid Length",
    0x0006: "Unsupported Protocol",
    0x0064: "Invalid Parameter",
    0x0065: "Invalid Session",
    0x0069: "Invalid Context",
    0x006A: "Invalid Reply",
    0x006B: "Invalid Command Length",
    0x006C: "Invalid Encapsulation Protocol",
    0x006D: "Insufficient Buffer",
    0x006E: "Unsupported Service",
    0x006F: "Attributes Not Settable",
    0x0070: "Privilege Violation",
    0x0071: "Device State Conflict",
    0x0072: "Invalid Reply Length",
    0x0073: "Partial Data",
    0x0074: "Connection ID Invalid",
    0x0075: "Invalid Sequence Number",
    0x0076: "Close Connection",
    0x0077: "Connection Abandoned",
    0x0078: "Invalid Network Path",
    0x0079: "Invalid Segment Type",
    0x007A: "Connection Not Found",
    0x007B: "Invalid Network Segment",
    0x007C: "Connection Already Exists",
}

# Protocol Constants
ENIP_HEADER_SIZE = 24

# ENIP Commands that contain data payload
ENIP_DATA_COMMANDS = {
    0x006F,  # SendRRData
    0x0070,  # SendUnitData
}

# Helper function to get command name
def get_enip_command_name(command_code: int) -> str:
    """Get ENIP command name from command code."""
    return ENIP_COMMANDS.get(command_code, f"Unknown_Command_0x{command_code:04X}")


# Helper function to get status description
def get_enip_status_description(status_code: int) -> str:
    """Get ENIP status description from status code."""
    return ENIP_STATUS_CODES.get(status_code, f"Unknown_Status_0x{status_code:04X}")


def detect_endianness(tcp_payload: bytes) -> str:
    """
    Detect endianness by checking which interpretation makes more sense.
    
    Args:
        tcp_payload: TCP payload bytes
        
    Returns:
        'big' or 'little' based on which interpretation is more reasonable
    """
    if len(tcp_payload) < 12:  # Need at least 12 bytes for basic validation
        return None  # Default to big-endian
    
    # Try big-endian first
    try:
        command_big = int.from_bytes(tcp_payload[0:2], byteorder='big')
        length_big = int.from_bytes(tcp_payload[2:4], byteorder='big')
        status_big = int.from_bytes(tcp_payload[8:12], byteorder='big')
        
        # Try little-endian
        command_little = int.from_bytes(tcp_payload[0:2], byteorder='little')
        length_little = int.from_bytes(tcp_payload[2:4], byteorder='little')
        status_little = int.from_bytes(tcp_payload[8:12], byteorder='little')
        
        # Score big-endian interpretation
        big_score = 0
        if command_big in ENIP_COMMANDS:
            big_score += 2  # Known command
        if command_big in ENIP_DATA_COMMANDS:
            big_score += 1  # Data command
        if 0 <= length_big <= 1024:  # Reasonable length (0-1KB)
            big_score += 1
        if status_big in ENIP_STATUS_CODES:
            big_score += 2  # Known status
        
        # Score little-endian interpretation
        little_score = 0
        if command_little in ENIP_COMMANDS:
            little_score += 2
        if command_little in ENIP_DATA_COMMANDS:
            little_score += 1
        if 0 <= length_little <= 1024:  # Reasonable length (0-1KB)
            little_score += 1
        if status_little in ENIP_STATUS_CODES:
            little_score += 2
        
        # Return the better interpretation
        return 'big' if big_score >= little_score else 'little'
        
    except Exception:
        return None  # Default to big-endian on error


def parse_enip_header(tcp_payload: bytes) -> dict:
    """
    Parse EtherNet/IP header from TCP payload with automatic byte order detection.
    
    Args:
        tcp_payload: TCP payload bytes
        
    Returns:
        Dictionary with parsed ENIP header fields
    """
    if len(tcp_payload) < ENIP_HEADER_SIZE:
        return None
    
    try:
        # Detect endianness automatically
        endianness = detect_endianness(tcp_payload)
        if endianness is None:
            return None
        
        enip_command = int.from_bytes(tcp_payload[0:2], byteorder=endianness)
        enip_length = int.from_bytes(tcp_payload[2:4], byteorder=endianness)
        enip_session_handle = int.from_bytes(tcp_payload[4:8], byteorder=endianness)
        enip_status = int.from_bytes(tcp_payload[8:12], byteorder=endianness)
        enip_sender_context = tcp_payload[12:20]
        enip_options = int.from_bytes(tcp_payload[20:24], byteorder=endianness)
        
        enip_header = {
            'protocol': 'ENIP',
            'command': enip_command,
            'command_name': get_enip_command_name(enip_command),
            'length': enip_length,
            'session_handle': enip_session_handle,
            'status': enip_status,
            'status_description': get_enip_status_description(enip_status),
            'sender_context': enip_sender_context,
            'options': enip_options,
            'is_data_command': enip_command in ENIP_DATA_COMMANDS,
            'endianness': endianness  # Include detected endianness in result
        }
        return enip_header
    except Exception as e:
        return None


def extract_enip_payload(tcp_payload: bytes, enip_info: dict) -> bytes:
    """
    Extract ENIP payload from TCP payload.
    
    Args:
        tcp_payload: TCP payload bytes
        enip_info: Parsed ENIP header information
        
    Returns:
        ENIP payload bytes or empty bytes if extraction fails
    """
    try:
        if len(tcp_payload) < ENIP_HEADER_SIZE:
            return b''
        
        if not enip_info:
            return b''
        
        # Get command-specific data length from ENIP header
        command_data_length = enip_info.get('length', 0)
        
        # Check if there's command-specific data
        if command_data_length == 0:
            # No command-specific data
            return b''
        
        if len(tcp_payload) < ENIP_HEADER_SIZE + command_data_length:
            return b''
        
        # Extract ENIP payload (command-specific data)
        enip_payload = tcp_payload[ENIP_HEADER_SIZE:ENIP_HEADER_SIZE + command_data_length]
        return enip_payload
        
    except Exception as e:
        return b''


# Example usage
if __name__ == "__main__":
    # Test ENIP-specific dictionaries and functions
    print("ENIP Command Examples:")
    test_commands = [0x0065, 0x0066, 0x0067, 0x006C, 0x006D]
    for cmd in test_commands:
        print(f"  0x{cmd:04X}: {get_enip_command_name(cmd)}")
    
    print("\nENIP Status Code Examples:")
    test_statuses = [0x0000, 0x0001, 0x0004, 0x006A]
    for status in test_statuses:
        print(f"  0x{status:04X}: {get_enip_status_description(status)}")
    
    print(f"\nENIP Protocol Constants:")
    print(f"  ENIP Header Size: {ENIP_HEADER_SIZE} bytes")
    
    print(f"\nENIP Data Commands: {ENIP_DATA_COMMANDS}")
    
    # Test ENIP header parsing
    print(f"\nTesting ENIP Header Parsing:")
    sample_enip_header = b'\x66\x00\x00\x0c\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    parsed = parse_enip_header(sample_enip_header + b'\x00' * 10)  # Add some padding
    if parsed:
        print(f"  Command: {parsed['command_name']} (0x{parsed['command']:04X})")
        print(f"  Length: {parsed['length']}")
        print(f"  Status: {parsed['status_description']}")
        print(f"  Is Data Command: {parsed['is_data_command']}")
        print(f"  Detected Endianness: {parsed['endianness']}")
    else:
        print("  Failed to parse header")
    
    # Test ENIP payload extraction
    print(f"\nTesting ENIP Payload Extraction:")
    if parsed:
        enip_payload = extract_enip_payload(sample_enip_header + b'\x00' * 10, parsed)
        print(f"  ENIP Payload Length: {len(enip_payload)} bytes")
        print(f"  ENIP Payload Hex: {enip_payload.hex()}")