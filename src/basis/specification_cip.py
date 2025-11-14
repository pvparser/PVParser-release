# CIP (Common Industrial Protocol) Specification
# Based on CIP Common Specification and vendor implementations

# CPF (Common Packet Format) Item Types
CPF_ITEM_TYPES = {
    0x0000: "Null Address Item",
    0x00A1: "Connection Based Address Item",
    0x00B1: "Connected Transport Packet",
    0x00B2: "Unconnected Data Item", 
    0x00B3: "List Identity Response",
    0x8000: "Sockaddr Info O->T",
    0x8001: "Sockaddr Info T->O",
    0x8002: "Sequenced Address Item"
}

# CIP Object Class Codes (commonly used with ENIP)
CIP_OBJECT_CLASSES = {
    0x01: "Identity Object",
    0x02: "Message Router Object",
    0x03: "DeviceNet Object",
    0x04: "Assembly Object",
    0x05: "Connection Object",
    0x06: "Connection Manager Object",
    0x07: "Register Object",
    0x08: "Discrete Input Point Object",
    0x09: "Discrete Output Point Object",
    0x0A: "Analog Input Point Object",
    0x0B: "Analog Output Point Object",
    0x0E: "Presence Sensing Object",
    0x0F: "Parameter Object",
    0x10: "Parameter Group Object",
    0x12: "Group Object",
    0x1D: "Selection Object",
    0x1E: "Connection Configuration Object",
    0x23: "Position Sensor Object",
    0x24: "Position Controller Supervisor Object",
    0x25: "Position Controller Object",
    0x26: "Block Sequencer Object",
    0x27: "Command Block Object",
    0x28: "Motor Data Object",
    0x29: "Control Supervisor Object",
    0x2A: "AC/DC Drive Object",
    0x2B: "Acknowledge Handler Object",
    0x2C: "Overload Object",
    0x2D: "Softstart Object",
    0x2E: "Selection Object",
    0x30: "S-Device Supervisor Object",
    0x31: "S-Analog Sensor Object",
    0x32: "S-Analog Actuator Object",
    0x33: "S-Single Stage Controller Object",
    0x34: "S-Gas Calibration Object",
    0x35: "Trip Point Object",
    0x37: "File Object",
    0x38: "S-Partial Pressure Object",
    0x39: "Safety Supervisor Object",
    0x3A: "Safety Validator Object",
    0x3B: "Safety Discrete Output Point Object",
    0x3C: "Safety Discrete Output Group Object",
    0x3D: "Safety Discrete Input Point Object",
    0x3E: "Safety Dual Channel Output Object",
    0x3F: "S-Sensor Calibration Object",
    0x40: "Event Log Object",
    0x41: "Motion Device Axis Object",
    0x42: "Time Sync Object",
    0x43: "Modbus Object",
    0x44: "Originator Connection List Object",
    0x45: "Target Connection List Object",
    0x46: "Connection Status Object",
    0x47: "Base Energy Object",
    0x48: "Electrical Energy Object",
    0x49: "Non-Electrical Energy Object",
    0x4A: "Base Switch Object",
    0x4B: "SNMP Object",
    0x4C: "Power Management Object",
    0x4D: "RSTP Bridge Object",
    0x4E: "RSTP Port Object",
    0x4F: "PRP/HSR Protocol Object",
    0x50: "PRP/HSR Nodes Table Object",
    0xF0: "ControlNet Object",
    0xF1: "ControlNet Keeper Object",
    0xF2: "ControlNet Scheduling Object",
    0xF3: "Connection Configuration Object",
    0xF4: "Port Object",
    0xF5: "TCP/IP Interface Object",
    0xF6: "Ethernet Link Object",
    0xF7: "CompoNet Object",
    0xF8: "CompoNet Repeater Object",
}

# Complete CIP Service Codes Dictionary
# Based on CIP Common Specification and vendor implementations
CIP_SERVICE_CODES = {
    # Standard CIP Services (0x01-0x1F)
    0x01: "Get_Attributes_All",              # Get All Attributes
    0x02: "Set_Attributes_All",              # Set All Attributes  
    0x03: "Get_Attribute_List",              # Get Attribute List
    0x04: "Set_Attribute_List",              # Set Attribute List
    0x05: "Reset",                           # Reset
    0x06: "Start",                           # Start
    0x07: "Stop",                            # Stop
    0x08: "Create",                          # Create
    0x09: "Delete",                          # Delete
    0x0A: "Multiple_Service_Packet",         # Multiple Service Packet
    0x0B: "Apply_Attributes",                # Apply Attributes
    0x0C: "Get_Attribute_Single",            # Get Single Attribute (corrected from 0x0E)
    0x0D: "Set_Attribute_Single",            # Set Single Attribute (corrected from 0x10)
    0x0E: "Find_Next_Object_Instance",       # Find Next Object Instance
    0x0F: "Error_Response",                  # Error Response (when high bit set)
    0x10: "Restore",                         # Restore
    0x11: "Save",                            # Save
    0x12: "NOP",                             # No Operation
    0x13: "Get_Member",                      # Get Member
    0x14: "Set_Member",                      # Set Member
    0x15: "Insert_Member",                   # Insert Member
    0x16: "Remove_Member",                   # Remove Member
    0x17: "Group_Sync",                      # Group Sync
    0x18: "Get_Attributes_Single_Reply",     # Get Attributes Single Reply
    0x19: "Set_Attributes_Single_Reply",     # Set Attributes Single Reply
    0x1A: "Get_Attribute_List_Reply",        # Get Attribute List Reply
    0x1B: "Set_Attribute_List_Reply",        # Set Attribute List Reply
    0x1C: "Get_Member_Reply",                # Get Member Reply
    0x1D: "Set_Member_Reply",                # Set Member Reply
    0x1E: "Insert_Member_Reply",             # Insert Member Reply
    0x1F: "Remove_Member_Reply",             # Remove Member Reply

    # Extended Services (0x20-0x3F)
    0x20: "Get_Attribute_Single_Extended",   # Extended Get Single Attribute
    0x21: "Set_Attribute_Single_Extended",   # Extended Set Single Attribute
    0x22: "Find_Next_Object_Extended",       # Extended Find Next Object
    0x23: "Restore_Extended",                # Extended Restore
    0x24: "Save_Extended",                   # Extended Save
    0x25: "Get_Member_Extended",             # Extended Get Member
    0x26: "Set_Member_Extended",             # Extended Set Member
    0x27: "Create_Extended",                 # Extended Create
    0x28: "Delete_Extended",                 # Extended Delete
    0x29: "Get_Attributes_All_Extended",     # Extended Get All Attributes
    0x2A: "Set_Attributes_All_Extended",     # Extended Set All Attributes
    0x2B: "Get_Attribute_List_Extended",     # Extended Get Attribute List
    0x2C: "Set_Attribute_List_Extended",     # Extended Set Attribute List
    0x2D: "Reset_Extended",                  # Extended Reset
    0x2E: "Start_Extended",                  # Extended Start
    0x2F: "Stop_Extended",                   # Extended Stop

    # DeviceNet/ControlNet Services (0x30-0x47)
    0x30: "Kick_Timer",                      # DeviceNet Kick Timer
    0x31: "Open_Connection",                 # DeviceNet Open Connection
    0x32: "Close_Connection",                # DeviceNet Close Connection
    0x33: "Stop_Connection",                 # DeviceNet Stop Connection
    0x34: "Change_Start_Stop",               # DeviceNet Change Start/Stop
    0x35: "Get_Connection_Data",             # DeviceNet Get Connection Data
    0x36: "Search_Connection_Data",          # DeviceNet Search Connection Data
    0x37: "Get_Connection_Owner",            # DeviceNet Get Connection Owner
    0x38: "Reserved_38",                     # Reserved
    0x39: "Reserved_39",                     # Reserved
    0x3A: "Reserved_3A",                     # Reserved
    0x3B: "Reserved_3B",                     # Reserved
    0x3C: "Reserved_3C",                     # Reserved
    0x3D: "Reserved_3D",                     # Reserved
    0x3E: "Reserved_3E",                     # Reserved
    0x3F: "Reserved_3F",                     # Reserved

    # ControlLogix/PLC Services (0x40-0x5F)
    0x40: "Read_Tag",                        # Read Tag (ControlLogix)
    0x41: "Write_Tag",                       # Write Tag (ControlLogix)
    0x42: "Read_Tag_Fragmented",             # Read Tag Fragmented
    0x43: "Write_Tag_Fragmented",            # Write Tag Fragmented
    0x44: "Read_Modify_Write_Tag",           # Read Modify Write Tag
    0x45: "Read_Tag_Service_Reply",          # Read Tag Service Reply
    0x46: "Write_Tag_Service_Reply",         # Write Tag Service Reply
    0x47: "Read_Tag_Fragmented_Reply",       # Read Tag Fragmented Reply
    0x48: "Write_Tag_Fragmented_Reply",      # Write Tag Fragmented Reply
    0x49: "Read_Modify_Write_Tag_Reply",     # Read Modify Write Tag Reply
    0x4A: "Reserved_4A",                     # Reserved
    0x4B: "Execute_PCCC",                    # Execute PCCC (PLC-5/SLC500)
    0x4C: "Read_Tag_Service",                # Read Tag Service (AB specific)
    0x4D: "Write_Tag_Service",               # Write Tag Service (AB specific)
    0x4E: "Read_Template",                   # Read Template
    0x4F: "Write_Template",                  # Write Template

    # Connection Manager Services (0x50-0x5F)
    0x50: "Unconnected_Send",                # Unconnected Send
    0x51: "Forward_Close",                   # Forward Close
    0x52: "Forward_Open",                    # Forward Open
    0x53: "Get_Connection_Data_CM",          # Connection Manager Get Connection Data
    0x54: "Search_Connection_Data_CM",       # Connection Manager Search Connection Data
    0x55: "Get_Connection_Owner_CM",         # Connection Manager Get Connection Owner
    0x56: "Large_Forward_Open",              # Large Forward Open
    0x57: "Get_Connection_Info",             # Get Connection Info
    0x58: "Get_Additional_Status",           # Get Additional Status
    0x59: "Change_Socket_Address",           # Change Socket Address
    0x5A: "Get_Socket_Address_Info",         # Get Socket Address Info
    0x5B: "Reserved_5B",                     # Reserved
    0x5C: "Reserved_5C",                     # Reserved
    0x5D: "Reserved_5D",                     # Reserved
    0x5E: "Reserved_5E",                     # Reserved
    0x5F: "Reserved_5F",                     # Reserved

    # File Object Services (0x60-0x6F)
    0x60: "File_Open",                       # File Open
    0x61: "File_Read",                       # File Read
    0x62: "File_Write",                      # File Write
    0x63: "File_Close",                      # File Close
    0x64: "File_Delete",                     # File Delete
    0x65: "File_Rename",                     # File Rename
    0x66: "File_Copy",                       # File Copy
    0x67: "File_Move",                       # File Move
    0x68: "File_Truncate",                   # File Truncate
    0x69: "File_Get_Info",                   # File Get Info
    0x6A: "File_Change_Mode",                # File Change Mode
    0x6B: "File_Change_Owner",               # File Change Owner
    0x6C: "Directory_Open",                  # Directory Open
    0x6D: "Directory_Read",                  # Directory Read
    0x6E: "Directory_Close",                 # Directory Close
    0x6F: "Directory_Create",                # Directory Create

    # Motion Services (0x70-0x7F)
    0x70: "Motion_Set_Attribute",            # Motion Set Attribute
    0x71: "Motion_Get_Attribute",            # Motion Get Attribute
    0x72: "Motion_Start_Stop_List",          # Motion Start Stop List
    0x73: "Motion_Device_Control",           # Motion Device Control
    0x74: "Motion_Group_Sync",               # Motion Group Sync
    0x75: "Motion_Coordinate_System",        # Motion Coordinate System
    0x76: "Motion_Cam_Profile",              # Motion Cam Profile
    0x77: "Motion_Event_Action",             # Motion Event Action
    0x78: "Motion_Time_Sync",                # Motion Time Sync
    0x79: "Motion_Status_Report",            # Motion Status Report
    0x7A: "Motion_Configuration",            # Motion Configuration
    0x7B: "Motion_Information",              # Motion Information
    0x7C: "Motion_Command",                  # Motion Command
    0x7D: "Motion_Response",                 # Motion Response
    0x7E: "Motion_Reserved_7E",              # Reserved
    0x7F: "Motion_Reserved_7F",              # Reserved

    # Safety Services (0x80-0x8F)
    0x80: "Safety_Open",                     # Safety Open
    0x81: "Safety_Close",                    # Safety Close
    0x82: "Safety_Data",                     # Safety Data
    0x83: "Safety_Ack",                      # Safety Acknowledgment
    0x84: "Safety_Status",                   # Safety Status
    0x85: "Safety_Configuration",            # Safety Configuration
    0x86: "Safety_Reset",                    # Safety Reset
    0x87: "Safety_Test",                     # Safety Test
    0x88: "Safety_Enable",                   # Safety Enable
    0x89: "Safety_Disable",                  # Safety Disable
    0x8A: "Safety_Reserved_8A",              # Reserved
    0x8B: "Safety_Reserved_8B",              # Reserved
    0x8C: "Safety_Reserved_8C",              # Reserved
    0x8D: "Safety_Reserved_8D",              # Reserved
    0x8E: "Safety_Reserved_8E",              # Reserved
    0x8F: "Safety_Reserved_8F",              # Reserved

    # Time Sync Services (0x90-0x9F)
    0x90: "Time_Sync_Set",                   # Time Sync Set
    0x91: "Time_Sync_Get",                   # Time Sync Get
    0x92: "Time_Sync_Apply",                 # Time Sync Apply
    0x93: "Time_Sync_Status",                # Time Sync Status
    0x94: "Time_Sync_Adjust",                # Time Sync Adjust
    0x95: "Time_Sync_Reserved_95",           # Reserved
    0x96: "Time_Sync_Reserved_96",           # Reserved
    0x97: "Time_Sync_Reserved_97",           # Reserved
    0x98: "Time_Sync_Reserved_98",           # Reserved
    0x99: "Time_Sync_Reserved_99",           # Reserved
    0x9A: "Time_Sync_Reserved_9A",           # Reserved
    0x9B: "Time_Sync_Reserved_9B",           # Reserved
    0x9C: "Time_Sync_Reserved_9C",           # Reserved
    0x9D: "Time_Sync_Reserved_9D",           # Reserved
    0x9E: "Time_Sync_Reserved_9E",           # Reserved
    0x9F: "Time_Sync_Reserved_9F",           # Reserved

    # Vendor Specific Services (0xA0-0xFF)
    0xA0: "Vendor_Specific_A0",              # Vendor Specific
    0xA1: "Vendor_Specific_A1",              # Vendor Specific
    0xA2: "Vendor_Specific_A2",              # Vendor Specific
    0xA3: "Vendor_Specific_A3",              # Vendor Specific
    0xA4: "Vendor_Specific_A4",              # Vendor Specific
    0xA5: "Vendor_Specific_A5",              # Vendor Specific
    0xA6: "Vendor_Specific_A6",              # Vendor Specific
    0xA7: "Vendor_Specific_A7",              # Vendor Specific
    0xA8: "Vendor_Specific_A8",              # Vendor Specific
    0xA9: "Vendor_Specific_A9",              # Vendor Specific
    0xAA: "Modbus_Read_Discrete_Inputs",     # Modbus Read Discrete Inputs
    0xAB: "Modbus_Read_Coils",               # Modbus Read Coils
    0xAC: "Modbus_Read_Input_Registers",     # Modbus Read Input Registers
    0xAD: "Modbus_Read_Holding_Registers",   # Modbus Read Holding Registers
    0xAE: "Modbus_Write_Single_Coil",        # Modbus Write Single Coil
    0xAF: "Modbus_Write_Single_Register",    # Modbus Write Single Register
    0xB0: "Modbus_Write_Multiple_Coils",     # Modbus Write Multiple Coils
    0xB1: "Modbus_Write_Multiple_Registers", # Modbus Write Multiple Registers
    0xB2: "Modbus_Passthrough",              # Modbus Passthrough
    0xB3: "Get_Attribute_All_Response",      # Response with error bit set
    0xB4: "Set_Attribute_All_Response",      # Response with error bit set
    0xB5: "Vendor_Specific_B5",              # Vendor Specific
    0xB6: "Vendor_Specific_B6",              # Vendor Specific
    0xB7: "Vendor_Specific_B7",              # Vendor Specific
    0xB8: "Vendor_Specific_B8",              # Vendor Specific
    0xB9: "Vendor_Specific_B9",              # Vendor Specific
    0xBA: "Vendor_Specific_BA",              # Vendor Specific
    0xBB: "Vendor_Specific_BB",              # Vendor Specific
    0xBC: "Vendor_Specific_BC",              # Vendor Specific
    0xBD: "Vendor_Specific_BD",              # Vendor Specific
    0xBE: "Vendor_Specific_BE",              # Vendor Specific
    0xBF: "Vendor_Specific_BF",              # Vendor Specific

    # Allen-Bradley Specific Services (Common vendor extensions)
    0xC0: "AB_Read_Program_Header",          # AB: Read Program Header
    0xC1: "AB_Read_Program_Data",            # AB: Read Program Data
    0xC2: "AB_Write_Program_Data",           # AB: Write Program Data
    0xC3: "AB_Download_Program",             # AB: Download Program
    0xC4: "AB_Upload_Program",               # AB: Upload Program
    0xC5: "AB_Get_Module_Info",              # AB: Get Module Info
    0xC6: "AB_Set_Module_Config",            # AB: Set Module Config
    0xC7: "AB_Get_Status_Data",              # AB: Get Status Data
    0xC8: "AB_Set_Status_Data",              # AB: Set Status Data
    0xC9: "AB_Execute_Protected_Function",   # AB: Execute Protected Function
    0xCA: "AB_Get_Protected_Data",           # AB: Get Protected Data
    0xCB: "AB_Set_Protected_Data",           # AB: Set Protected Data
    0xCC: "Read_Tag_Service",                # Corrected: AB Read Tag Service
    0xCD: "Write_Tag_Service",               # Corrected: AB Write Tag Service
    0xCE: "AB_Get_Instance_Attributes",      # AB: Get Instance Attributes
    0xCF: "AB_Set_Instance_Attributes",      # AB: Set Instance Attributes

    # Additional Vendor Services
    0xD0: "Schneider_Specific_D0",           # Schneider Electric Specific
    0xD1: "Siemens_Specific_D1",             # Siemens Specific
    0xD2: "Omron_Specific_D2",               # Omron Specific
    0xD3: "Mitsubishi_Specific_D3",          # Mitsubishi Specific
    0xD4: "GE_Specific_D4",                  # GE Specific
    0xD5: "Emerson_Specific_D5",             # Emerson Specific
    0xD6: "Honeywell_Specific_D6",           # Honeywell Specific
    0xD7: "Yokogawa_Specific_D7",            # Yokogawa Specific
    0xD8: "Vendor_Specific_D8",              # Generic Vendor Specific
    0xD9: "Vendor_Specific_D9",              # Generic Vendor Specific
    0xDA: "Vendor_Specific_DA",              # Generic Vendor Specific
    0xDB: "Vendor_Specific_DB",              # Generic Vendor Specific
    0xDC: "Vendor_Specific_DC",              # Generic Vendor Specific
    0xDD: "Vendor_Specific_DD",              # Generic Vendor Specific
    0xDE: "Vendor_Specific_DE",              # Generic Vendor Specific
    0xDF: "Vendor_Specific_DF",              # Generic Vendor Specific

    # High-bit set indicates response (0x80 | original_service)
    0xE0: "Response_E0",                     # Response to 0x60
    0xE1: "Response_E1",                     # Response to 0x61
    0xE2: "Response_E2",                     # Response to 0x62
    0xE3: "Response_E3",                     # Response to 0x63
    0xE4: "Response_E4",                     # Response to 0x64
    0xE5: "Response_E5",                     # Response to 0x65
    0xE6: "Response_E6",                     # Response to 0x66
    0xE7: "Response_E7",                     # Response to 0x67
    0xE8: "Response_E8",                     # Response to 0x68
    0xE9: "Response_E9",                     # Response to 0x69
    0xEA: "Response_EA",                     # Response to 0x6A
    0xEB: "Response_EB",                     # Response to 0x6B
    0xEC: "Response_EC",                     # Response to 0x6C
    0xED: "Response_ED",                     # Response to 0x6D
    0xEE: "Response_EE",                     # Response to 0x6E
    0xEF: "Response_EF",                     # Response to 0x6F
    0xF0: "Reserved_F0",                     # Reserved
    0xF1: "Reserved_F1",                     # Reserved
    0xF2: "Reserved_F2",                     # Reserved
    0xF3: "Reserved_F3",                     # Reserved
    0xF4: "Reserved_F4",                     # Reserved
    0xF5: "Reserved_F5",                     # Reserved
    0xF6: "Reserved_F6",                     # Reserved
    0xF7: "Reserved_F7",                     # Reserved
    0xF8: "Reserved_F8",                     # Reserved
    0xF9: "Reserved_F9",                     # Reserved
    0xFA: "Reserved_FA",                     # Reserved
    0xFB: "Reserved_FB",                     # Reserved
    0xFC: "Reserved_FC",                     # Reserved
    0xFD: "Reserved_FD",                     # Reserved
    0xFE: "Reserved_FE",                     # Reserved
    0xFF: "Reserved_FF",                     # Reserved
}

# CIP Error Codes (General Status Codes)
CIP_ERROR_CODES = {
    0x00: "Success",
    0x01: "Connection failure",
    0x02: "Resource unavailable", 
    0x03: "Invalid parameter value",
    0x04: "Path segment error",
    0x05: "Path destination unknown",
    0x06: "Partial transfer",
    0x07: "Connection lost",
    0x08: "Service not supported",
    0x09: "Invalid attribute value",
    0x0A: "Attribute list error",
    0x0B: "Already in requested mode/state",
    0x0C: "Object state conflict",
    0x0D: "Object already exists",
    0x0E: "Attribute not settable",
    0x0F: "Privilege violation",
    0x10: "Device state conflict",
    0x11: "Reply data too large",
    0x12: "Fragmentation of a primitive value",
    0x13: "Not enough data",
    0x14: "Attribute not supported",
    0x15: "Too much data",
    0x16: "Object does not exist",
    0x17: "Service fragmentation sequence not in progress",
    0x18: "No stored attribute data",
    0x19: "Store operation failure",
    0x1A: "Routing failure, request packet too large",
    0x1B: "Routing failure, response packet too large",
    0x1C: "Missing attribute list entry data",
    0x1D: "Invalid attribute value list",
    0x1E: "Embedded service error",
    0x1F: "Vendor specific error",
    0x20: "Invalid parameter",
    0x21: "Write-once value or medium already written",
    0x22: "Invalid Reply Received",
    0x23: "Buffer Overflow",
    0x24: "Invalid Message Format",
    0x25: "Key Failure in path",
    0x26: "Path Size Invalid",
    0x27: "Unexpected attribute in list",
    0x28: "Invalid Member ID",
    0x29: "Member not settable",
    0x2A: "Group 2 only server general failure",
    0x2B: "Unknown Modbus error",
    0x2C: "Attribute not gettable",
}

# CIP Protocol Constants
CIP_MIN_HEADER_SIZE = 6

# Common CIP Services for Data (services that typically carry application data)
CIP_DATA_SERVICES = {
    0x0C,  # Get_Attribute_Single
    0x0D,  # Set_Attribute_Single
    0x4C,  # Read_Tag_Service (AB specific)
    0x4D,  # Write_Tag_Service (AB specific) 
    0x52,  # Forward_Open
    0x54,  # Forward_Close
    0x56,  # Unconnected_Send
}


# Helper functions
def get_cip_object_class_name(class_code: int) -> str:
    """Get CIP object class name from class code."""
    return CIP_OBJECT_CLASSES.get(class_code, f"Unknown_Class_0x{class_code:02X}")


def get_cip_service_name(service_code: int) -> str:
    """Get CIP service name from service code."""
    return CIP_SERVICE_CODES.get(service_code, f"Unknown_Service_0x{service_code:02X}")


def get_cip_error_description(error_code: int) -> str:
    """Get CIP error description from error code."""
    return CIP_ERROR_CODES.get(error_code, f"Unknown_Error_0x{error_code:02X}")


def parse_cpf_header(enip_payload: bytes, endianness: str = 'little') -> dict:
    """
    Parse CPF (Common Packet Format) header from ENIP payload.
    This is the data between ENIP header and CIP header.
    
    Args:
        enip_payload: ENIP payload bytes (data after ENIP header)
        endianness: Byte order ('little' or 'big')
        
    Returns:
        Dictionary containing CPF header information or None if parsing fails
    """
    try:
        if not enip_payload or len(enip_payload) < 8:
            return None
        
        # CPF Header format for SendRRData/SendUnitData:
        # [Interface Handle (4)] [Timeout (2)] [Item Count (2)] [Items...]
        
        interface_handle = int.from_bytes(enip_payload[0:4], byteorder=endianness)
        timeout = int.from_bytes(enip_payload[4:6], byteorder=endianness)
        item_count = int.from_bytes(enip_payload[6:8], byteorder=endianness)
        
        result = {
            'interface_handle': interface_handle,
            'timeout': timeout,
            'item_count': item_count,
            'items': [],
            'header_size': 8,  # Base header size
            'cip_data_offset': None,
            'cip_data_length': 0
        }
        
        # Parse items
        offset = 8
        cip_data_item_found = False
        
        for i in range(item_count):
            if offset + 4 > len(enip_payload):
                break
            
            item_type = int.from_bytes(enip_payload[offset:offset+2], byteorder=endianness)
            item_length = int.from_bytes(enip_payload[offset+2:offset+4], byteorder=endianness)
            
            item_info = {
                'index': i,
                'type': item_type,
                'type_name': CPF_ITEM_TYPES.get(item_type, f"Unknown (0x{item_type:04X})"),
                'length': item_length,
                'data_offset': offset + 4,
                'is_cip_data': item_type in [0x00B1, 0x00B2]
            }
            
            # Extract item data if present and handle sequence count for connected packets
            if item_length > 0 and offset + 4 + item_length <= len(enip_payload):
                raw_item_data = enip_payload[offset + 4:offset + 4 + item_length]
                
                if item_type == 0x00B1 and item_length >= 2:  # Connected Transport Packet
                    # Extract sequence count and actual CIP data
                    sequence_count = int.from_bytes(raw_item_data[0:2], byteorder=endianness)
                    item_info['sequence_count'] = sequence_count
                    item_info['data'] = raw_item_data[2:]  # CIP data starts after sequence count
                else:
                    item_info['data'] = raw_item_data
            else:
                item_info['data'] = b''
            
            # Check if this is a CIP data item
            if item_type in [0x00B1, 0x00B2] and not cip_data_item_found:
                if item_type == 0x00B1:  # Connected Transport Packet
                    # Connected packets have 2-byte sequence count before CIP data
                    result['cip_data_offset'] = offset + 4 + 2  # Skip item header + sequence count
                    result['cip_data_length'] = item_length - 2  # CIP data length minus sequence count
                    result['has_sequence_count'] = True
                else:  # 0x00B2 - Unconnected Data Item
                    # Unconnected packets start directly with CIP data
                    result['cip_data_offset'] = offset + 4  # Points to Service Code
                    result['cip_data_length'] = item_length  # Only the CIP data length
                    result['has_sequence_count'] = False
                cip_data_item_found = True
            
            result['items'].append(item_info)
            
            # Move to next item
            offset += 4 + item_length
            if offset > len(enip_payload):
                break
        
        # Update total header size (includes all items)
        result['total_size'] = offset
        
        return result
        
    except Exception as e:
        return None


def parse_cip_header(enip_payload: bytes, endianness: str = 'little') -> dict:
    """
    Parse CIP header by first parsing CPF header, then integrating CIP information.
    
    Args:
        enip_payload: ENIP payload bytes (data after ENIP header)
        endianness: Byte order ('little' or 'big')
        
    Returns:
        Dictionary containing integrated CPF and CIP header information or None if parsing fails
    """
    try:
        # First parse CPF header to get structure and locate CIP data
        cpf_info = parse_cpf_header(enip_payload, endianness)
        if not cpf_info:
            return None
        
        # Check if CIP data item was found
        if cpf_info['cip_data_offset'] is None:
            return None
        
        # Extract CIP data from the identified item
        cip_data = enip_payload[cpf_info['cip_data_offset']:cpf_info['cip_data_offset'] + cpf_info['cip_data_length']]
        
        if len(cip_data) < 2:
            return None
        
        # Parse CIP-specific header information
        service_code = cip_data[0]
        request_path_size = cip_data[1]  # Size in 16-bit words
        
        # Calculate basic CIP header size
        basic_cip_header_size = 2 + (request_path_size * 2)
        
        if len(cip_data) < basic_cip_header_size:
            return None
        
        # Extract request path
        request_path = cip_data[2:2 + (request_path_size * 2)]
        
        # Start building integrated result
        result = {
            'protocol': 'CIP',
            'endianness': endianness,
            # CPF information
            'interface_handle': cpf_info['interface_handle'],
            'timeout': cpf_info['timeout'],
            'item_count': cpf_info['item_count'],
            'cpf_items': cpf_info['items'],
            'cpf_header_size': cpf_info['header_size'],
            'cpf_total_size': cpf_info['total_size'],
            
            # CIP information
            'service_code': service_code,
            'service_name': get_cip_service_name(service_code),
            'request_path_size': request_path_size,
            'request_path': request_path,
            'is_response': bool(service_code & 0x80),
            'is_data_service': is_cip_data_service(service_code),
            
            # Data location
            'cip_data_offset': cpf_info['cip_data_offset'],
            'cip_data_length': cpf_info['cip_data_length'],
            'cip_header_size': basic_cip_header_size
        }
        
        # Handle response packets (with status information)
        if service_code & 0x80:  # Response packet
            if len(cip_data) >= basic_cip_header_size + 2:
                general_status = cip_data[basic_cip_header_size]
                additional_status_size = cip_data[basic_cip_header_size + 1]
                
                response_header_size = basic_cip_header_size + 2 + (additional_status_size * 2)
                
                result.update({
                    'general_status': general_status,
                    'additional_status_size': additional_status_size,
                    'cip_header_size': response_header_size
                })
                
                # Extract additional status if present
                if additional_status_size > 0 and len(cip_data) >= response_header_size:
                    additional_status_data = cip_data[basic_cip_header_size + 2:response_header_size]
                    additional_status = []
                    for i in range(0, len(additional_status_data), 2):
                        if i + 1 < len(additional_status_data):
                            status_word = int.from_bytes(additional_status_data[i:i+2], byteorder=endianness)
                            additional_status.append(status_word)
                    result['additional_status'] = additional_status
            else:
                result['cip_header_size'] = basic_cip_header_size
        else:  # Request packet
            result['cip_header_size'] = basic_cip_header_size
        
        # Calculate total header size (CPF + CIP)
        result['total_header_size'] = cpf_info['cip_data_offset'] + result['cip_header_size']
        
        return result
        
    except Exception as e:
        return None


def is_cip_data_service(service_code: int) -> bool:
    """
    Check if a CIP service code carries data payload.
    
    Args:
        service_code: CIP service code (with or without response bit)
        
    Returns:
        True if service carries data, False otherwise
    """
    # Remove response bit (bit 7) to get base service code
    base_service = service_code & 0x7F
    return base_service in CIP_DATA_SERVICES


def extract_pure_data_from_cip(enip_payload: bytes, cip_header_info: dict) -> bytes:
    """
    Extract pure data from ENIP payload by removing all headers (CPF + CIP).
    Only extracts data if the service code is in CIP_DATA_SERVICES list.
    
    Args:
        enip_payload: ENIP payload bytes (data after ENIP header)
        cip_header_info: CIP header information
        
    Returns:
        Pure data bytes if service is in CIP_DATA_SERVICES, otherwise empty bytes
    """
    try:
        if not enip_payload:
            return b''
        
        # Parse integrated CPF and CIP header information
        if not cip_header_info:
            return b''
        
        # Check if this service carries data payload
        service_code = cip_header_info['service_code']
        if not is_cip_data_service(service_code):
            return b''
        
        # Get total header size (CPF + CIP)
        total_header_size = cip_header_info.get('total_header_size', 0)
        
        # Validate that there's data after the total header
        if len(enip_payload) <= total_header_size:
            return b''  # No data after header
        
        # Return pure data part (skip all headers)
        return enip_payload[total_header_size:]
        
    except Exception as e:
        return b''


# Example usage
if __name__ == "__main__":
    print("\nCIP Object Class Examples:")
    test_classes = [0x01, 0x04, 0x06, 0xF5, 0xF6]
    for cls in test_classes:
        print(f"  0x{cls:02X}: {get_cip_object_class_name(cls)}")
        
    print("CIP Service Code Examples:")
    test_services = [0x01, 0x0C, 0x0D, 0x4C, 0x4D, 0x52, 0x54, 0xCC, 0xCD]
    for service in test_services:
        name = get_cip_service_name(service)
        print(f"  0x{service:02X}: {name}")
    
    print("\nCIP Error Code Examples:")
    test_errors = [0x00, 0x01, 0x04, 0x08, 0x16, 0x1F]
    for error in test_errors:
        print(f"  0x{error:02X}: {get_cip_error_description(error)}")
    
    print(f"\nCIP Protocol Constants:")
    print(f"  CIP Min Header Size: {CIP_MIN_HEADER_SIZE} bytes")
    print(f"  CIP Data Services: {CIP_DATA_SERVICES}")
    
    # Test CIP header parsing
    print(f"\nTesting CIP Header Parsing:")
    sample_cip_header = b'\x0C\x01\x20\x01\x24\x01\x00\x00\x00\x00'  # Get_Attribute_Single with path
    parsed = parse_cip_header(sample_cip_header)
    if parsed:
        print(f"  Service: {parsed['service_name']} (0x{parsed['service']:02X})")
        print(f"  Path Size: {parsed['path_size']}")
        print(f"  Header Size: {parsed['header_size']} bytes")
        print(f"  Is Data Service: {parsed['is_data_service']}")
        print(f"  Is Response: {parsed['is_response']}")
    else:
        print("  Failed to parse header")