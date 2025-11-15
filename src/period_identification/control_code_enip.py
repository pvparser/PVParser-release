import os
import sys
from scapy.all import rdpcap
from scapy.layers.inet import TCP
from scapy.packet import Raw
import struct

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basis.specification_enip import get_enip_command_name, get_enip_status_description
from basis.specification_cip import get_cip_service_name, get_cip_error_description, get_cip_object_class_name

import struct
from typing import List, Dict, Any, Optional


def extract_enip_control_code(packet, endian="little") -> Optional[Dict[str, Any]]:
    """
    Extract ENIP control code from packet.
    
    Args:
        packet: Scapy packet object
        endian: Byte order ('little', 'big', or 'native')
    
    Returns:
        Dictionary containing ENIP control code information or None if no valid ENIP data found
        {
            "Control_Code": int,
        }
    """
    if not packet.haslayer(TCP):
        return None
    
    behaviors = extract_enip_cip_behaviors(packet, endian)
    if behaviors:
        return behaviors[0]["Control_Code"]
    return None


def extract_enip_cip_behaviors(packet, endian="little") -> Optional[List[Dict[str, Any]]]:
    """
    Extract ENIP + CIP function behavior sequences from packets.
    
    Args:
        packet: Scapy packet object
        endian: Byte order ('little', 'big', or 'native')
        
    Returns:
        List of behavior dictionaries or None if no valid ENIP data found:
        [
            {
                "Control_Code": "0x006F:0x4C",  # ENIP:CIP format
                "ENIP_Command": 0x006F,
                "ENIP_Description": "SendRRData",
                "CIP_Service": 0x4C,
                "CIP_Description": "Read_Tag_Service",
                "Direction": "Request|Response",
                "Session_Handle": 0x12345678,
                "Context": bytes,
                "Sequence_Number": int
            },
            ...
        ]
    """
    if not packet.haslayer(TCP):
        return None
    tcp_payload = bytes(packet[TCP].payload)
    
    if len(tcp_payload) < 24:
        return None  # ENIP header is at least 24 bytes
    
    # Set endian prefix for struct format
    endian_map = {
        'big': '>',
        'little': '<',
        'native': '='
    }
    endian_prefix = endian_map.get(endian, '<')
    
    behaviors = []
    offset = 0
    
    # Process all ENIP messages in the TCP payload (there might be multiple)
    while offset + 24 <= len(tcp_payload):
        try:
            behavior = parse_single_enip_message(tcp_payload[offset:], endian_prefix)
            if behavior:
                behaviors.append(behavior)
                # Move to next message (if any)
                message_length = behavior.get('Message_Length', 0)
                offset += 24 + message_length  # ENIP header + data length
            else:
                break
        except Exception as e:
            print(f"Error parsing ENIP message at offset {offset}: {e}")
            break
    
    return behaviors if behaviors else None


def parse_single_enip_message(data: bytes, endian_prefix: str) -> Optional[Dict[str, Any]]:
    """Parse a single ENIP message from byte data."""
    if len(data) < 24:
        return None
    
    try:
        # Parse ENIP header (24 bytes)
        enip_header = struct.unpack(f"{endian_prefix}HHIIIHHI", data[0:24])
        
        enip_command = enip_header[0]        # Command (2 bytes)
        enip_length = enip_header[1]         # Length (2 bytes)  
        session_handle = enip_header[2]      # Session Handle (4 bytes)
        status = enip_header[3]              # Status (4 bytes)
        context = data[16:24]                # Sender Context (8 bytes)
        options = enip_header[6]             # Options (4 bytes)
        
        # Get command description
        enip_command_desc = get_enip_command_name(enip_command)
        
        # Initialize result
        result = {
            "Control_Code": f"0x{enip_command:04X}",
            "ENIP_Command": enip_command,
            "ENIP_Command_Description": enip_command_desc,
            "Message_Length": enip_length,
            "Session_Handle": session_handle,
            "Status": status,
            "Status_Description": get_enip_status_description(status),
            "Context": context.hex(),
            "Options": options
        }
        
        # Only parse CIP data for commands that contain CIP messages
        cip_carrying_commands = {0x006F, 0x0070}  # SendRRData, SendUnitData
        
        if enip_command in cip_carrying_commands and enip_length > 0:
            if len(data) >= 24 + enip_length:
                cip_data = data[24:24 + enip_length]
                cip_behaviors = parse_cip_data(cip_data, enip_command, endian_prefix)
                
                if cip_behaviors:
                    # Merge CIP info into result
                    for cip_behavior in cip_behaviors:
                        combined_result = result.copy()
                        combined_result.update(cip_behavior)
                        combined_result["Control_Code"] = f"0x{enip_command:04X}:0x{cip_behavior.get('CIP_Service', 0):02X}"
                    
                    return combined_result
        
        return result
        
    except Exception as e:
        print(f"Error parsing ENIP header: {e}")
        return None


def parse_cip_data(cip_data: bytes, enip_command: int, endian_prefix: str) -> List[Dict[str, Any]]:
    """Parse CIP data from ENIP payload."""
    if len(cip_data) < 1:
        return []
    
    behaviors = []
    
    try:
        if enip_command == 0x006F:  # SendRRData
            # SendRRData format: Interface Handle(4) + Timeout(2) + Item Count(2) + Items...
            if len(cip_data) < 8:
                return []
            
            interface_handle = struct.unpack(f"{endian_prefix}I", cip_data[0:4])[0]
            timeout = struct.unpack(f"{endian_prefix}H", cip_data[4:6])[0]
            item_count = struct.unpack(f"{endian_prefix}H", cip_data[6:8])[0]
            
            offset = 8
            for i in range(item_count):
                if offset + 4 > len(cip_data):
                    break
                
                item_type = struct.unpack(f"{endian_prefix}H", cip_data[offset:offset+2])[0]
                item_length = struct.unpack(f"{endian_prefix}H", cip_data[offset+2:offset+4])[0]
                offset += 4
                
                if item_type == 0x00B2:  # Unconnected Data Item
                    if offset + item_length <= len(cip_data) and item_length > 0:
                        cip_msg = cip_data[offset:offset + item_length]
                        cip_behavior = parse_cip_message(cip_msg, endian_prefix)
                        if cip_behavior:
                            cip_behavior.update({
                                "Interface_Handle": interface_handle,
                                "Timeout": timeout,
                                "Item_Type": item_type,
                                "Item_Length": item_length
                            })
                            behaviors.append(cip_behavior)
                
                offset += item_length
        
        elif enip_command == 0x0070:  # SendUnitData
            # SendUnitData format: Interface Handle(4) + Timeout(2) + Item Count(2) + Items...
            # Similar to SendRRData but for connected messages
            if len(cip_data) < 8:
                return []
            
            interface_handle = struct.unpack(f"{endian_prefix}I", cip_data[0:4])[0]
            timeout = struct.unpack(f"{endian_prefix}H", cip_data[4:6])[0]
            item_count = struct.unpack(f"{endian_prefix}H", cip_data[6:8])[0]
            
            offset = 8
            for i in range(item_count):
                if offset + 4 > len(cip_data):
                    break
                
                item_type = struct.unpack(f"{endian_prefix}H", cip_data[offset:offset+2])[0]
                item_length = struct.unpack(f"{endian_prefix}H", cip_data[offset+2:offset+4])[0]
                offset += 4
                
                if item_type == 0x00B1:  # Connected Data Item
                    if offset + item_length <= len(cip_data) and item_length > 2:
                        # Skip connection sequence number (2 bytes)
                        cip_msg = cip_data[offset+2:offset + item_length]
                        cip_behavior = parse_cip_message(cip_msg, endian_prefix)
                        if cip_behavior:
                            cip_behavior.update({
                                "Interface_Handle": interface_handle,
                                "Timeout": timeout,
                                "Item_Type": item_type,
                                "Item_Length": item_length,
                                "Connected": True
                            })
                            behaviors.append(cip_behavior)
                
                offset += item_length
    
    except Exception as e:
        print(f"Error parsing CIP data: {e}")
    
    return behaviors


def parse_cip_message(cip_msg: bytes, endian_prefix: str) -> Optional[Dict[str, Any]]:
    """Parse individual CIP message."""
    if len(cip_msg) < 1:
        return None
    
    try:
        # CIP Service code is the first byte
        service_code = cip_msg[0]
        is_response = (service_code & 0x80) != 0
        actual_service = service_code & 0x7F
        
        result = {
            "CIP_Service": service_code,
            "CIP_Service_Request": actual_service,
            "CIP_Service_Description": get_cip_service_name(service_code),
            "CIP_Direction": "Response" if is_response else "Request"
        }
        
        if is_response:
            # Response format: Service(1) + Reserved(1) + General Status(1) + Extended Status Size(1) + [Extended Status] + Data
            if len(cip_msg) >= 4:
                general_status = cip_msg[2]
                extended_status_size = cip_msg[3]
                
                result.update({
                    "CIP_Status": general_status,
                    "CIP_Status_Description": get_cip_error_description(general_status),
                    "Extended_Status_Size": extended_status_size
                })
                
                if extended_status_size > 0 and len(cip_msg) >= 4 + extended_status_size * 2:
                    extended_status = cip_msg[4:4 + extended_status_size * 2]
                    result["Extended_Status"] = extended_status.hex()
        else:
            # Request format: Service(1) + Request Path Size(1) + Request Path + Data
            if len(cip_msg) >= 2:
                path_size = cip_msg[1]  # Size in 16-bit words
                path_bytes = path_size * 2
                
                result.update({
                    "CIP_Path_Size": path_size,
                    "CIP_Path_Bytes": path_bytes
                })
                
                if len(cip_msg) >= 2 + path_bytes:
                    if path_bytes > 0:
                        path_data = cip_msg[2:2 + path_bytes]
                        result["CIP_Path"] = path_data.hex()
                        
                        # Try to parse common path elements
                        path_info = parse_cip_path(path_data, endian_prefix)
                        if path_info:
                            result.update(path_info)
                    
                    # Remaining data after path
                    data_offset = 2 + path_bytes
                    if len(cip_msg) > data_offset:
                        service_data = cip_msg[data_offset:]
                        result["CIP_Data"] = service_data.hex()
                        result["CIP_Data_Length"] = len(service_data)
        
        return result
        
    except Exception as e:
        print(f"Error parsing CIP message: {e}")
        return None


def parse_cip_path(path_data: bytes, endian_prefix: str) -> Optional[Dict[str, Any]]:
    """Parse CIP path segments."""
    if len(path_data) < 2:
        return None
    
    try:
        result = {}
        offset = 0
        
        while offset < len(path_data):
            if offset + 1 >= len(path_data):
                break
                
            segment_type = path_data[offset]
            
            # Logical Segment (0x20-0x2F)
            if (segment_type & 0xE0) == 0x20:
                logical_type = (segment_type >> 2) & 0x07
                logical_format = segment_type & 0x03
                
                if logical_type == 0:  # Class ID
                    if logical_format == 0:  # 8-bit
                        if offset + 1 < len(path_data):
                            class_id = path_data[offset + 1]
                            result["Class_ID"] = class_id
                            result["Class_Name"] = get_cip_object_class_name(class_id)
                            offset += 2
                    elif logical_format == 1:  # 16-bit
                        if offset + 3 < len(path_data):
                            class_id = struct.unpack(f"{endian_prefix}H", path_data[offset+2:offset+4])[0]
                            result["Class_ID"] = class_id
                            result["Class_Name"] = get_cip_object_class_name(class_id)
                            offset += 4
                
                elif logical_type == 1:  # Instance ID
                    if logical_format == 0:  # 8-bit
                        if offset + 1 < len(path_data):
                            instance_id = path_data[offset + 1]
                            result["Instance_ID"] = instance_id
                            offset += 2
                    elif logical_format == 1:  # 16-bit
                        if offset + 3 < len(path_data):
                            instance_id = struct.unpack(f"{endian_prefix}H", path_data[offset+2:offset+4])[0]
                            result["Instance_ID"] = instance_id
                            offset += 4
                
                elif logical_type == 2:  # Attribute ID
                    if logical_format == 0:  # 8-bit
                        if offset + 1 < len(path_data):
                            attribute_id = path_data[offset + 1]
                            result["Attribute_ID"] = attribute_id
                            offset += 2
                    elif logical_format == 1:  # 16-bit
                        if offset + 3 < len(path_data):
                            attribute_id = struct.unpack(f"{endian_prefix}H", path_data[offset+2:offset+4])[0]
                            result["Attribute_ID"] = attribute_id
                            offset += 4
                else:
                    # Unknown logical segment
                    offset += 2
            else:
                # Other segment types - skip for now
                offset += 2
        
        return result if result else None
        
    except Exception as e:
        print(f"Error parsing CIP path: {e}")
        return None


# Example usage
if __name__ == "__main__":
    pcap_file = "dataset/swat/network/Dec2019_00000_20191206100500_00000w_filtered(20-100).pcap"
    packets = rdpcap(pcap_file)
    pkt_index = 0
    for pkt in packets:
        control_code = extract_enip_control_code(pkt)
        print(f"Packet {pkt_index}: {control_code}")
        pkt_index += 1
    
