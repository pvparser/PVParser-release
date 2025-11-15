import os
import struct
import sys
from typing import Optional, Dict, Any
from scapy.all import rdpcap, Packet, TCP, Raw

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basis.ics_basis import is_ics_port_by_protocol
from basis.specification_modbus import get_modbus_function_name, get_modbus_exception_name   


def extract_modbus_payload(packet) -> Optional[bytes]:
    if packet.haslayer(TCP):
        tcp_layer = packet[TCP]
        if is_ics_port_by_protocol(tcp_layer.sport, "modbus") or is_ics_port_by_protocol(tcp_layer.dport, "modbus"):
            if packet.haslayer(Raw):
                return bytes(packet[Raw].load)
            elif hasattr(tcp_layer, 'payload') and tcp_layer.payload:
                return bytes(tcp_layer.payload)
    
    return None


def parse_mbap_header(data: bytes) -> Optional[Dict[str, Any]]:
    """Parse Modbus MBAP header"""
    if len(data) < 7:
        return None
    
    try:
        transaction_id, protocol_id, length, unit_id = struct.unpack('>HHHB', data[:7])
        
        # Validate protocol ID (should be 0 for Modbus TCP)
        if protocol_id != 0:
            return None
        
        return {
            'transaction_id': transaction_id,
            'protocol_id': protocol_id,
            'length': length,
            'unit_id': unit_id
        }
    except struct.error:
        return None


def parse_function_code(data: bytes, offset: int = 7) -> Optional[Dict[str, Any]]:
    """Parse function code from Modbus PDU"""
    if len(data) <= offset:
        return None
    
    function_code = data[offset]
    
    # Check if this is an exception response
    is_exception = function_code >= 0x80
    actual_function_code = function_code & 0x7F
    
    result = {
        'function_code': function_code,
        'actual_function_code': actual_function_code,
        'function_name': get_modbus_function_name(actual_function_code),
        'is_exception': is_exception
    }
    
    # Parse exception code if present
    if is_exception and len(data) > offset + 1:
        exception_code = data[offset + 1]
        result['exception_code'] = exception_code
        result['exception_name'] = get_modbus_exception_name(exception_code)
    
    return result


def extract_modbus_control_code(packet) -> Dict[str, Any]:
    """
    Extract Modbus control code (function code) from scapy packet
    
    Args:
        packet: Scapy packet containing Modbus TCP data
        
    Returns:
        Dictionary containing extracted Modbus information:
        - transaction_id: Modbus transaction ID
        - unit_id: Modbus unit ID
        - function_code: Raw function code
        - actual_function_code: Function code without exception bit
        - function_name: Human readable function name
        - is_exception: Boolean indicating if this is an exception response
        - exception_code: Exception code (if applicable)
        - exception_name: Human readable exception name (if applicable)
        - parsed_data: Function-specific parsed data (if available)
        - src_port: Source TCP port
        - dst_port: Destination TCP port
    """
    # Extract Modbus payload from packet
    modbus_data = extract_modbus_payload(packet)
    if not modbus_data:
        return None
    
    # Parse MBAP header
    mbap_header = parse_mbap_header(modbus_data)
    if not mbap_header:
        return None
    
    # Parse function code
    function_info = parse_function_code(modbus_data)
    if not function_info:
        return None
    
    # Build result
    result = {
        'transaction_id': mbap_header['transaction_id'],
        'unit_id': mbap_header['unit_id'],
        'function_code': function_info['function_code'],
        'actual_function_code': function_info['actual_function_code'],
        'function_name': function_info['function_name'],
        'is_exception': function_info['is_exception']
    }
    
    if result["function_code"]:
        return result["function_code"]
    
    return None


if __name__ == "__main__":
    pcap_file = "dataset/scada/Modbus_polling_only_6RTU(106).pcap"
    packets = rdpcap(pcap_file)
    pkt_index = 0
    for pkt in packets:
        control_code = extract_modbus_control_code(pkt)
        print(f"Packet {pkt_index}: {control_code}")
        pkt_index += 1