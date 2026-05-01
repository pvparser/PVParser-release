import os
import sys
from typing import Optional

from scapy.all import TCP, Raw

# Import MMS specification functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basis.specification_mms import skip_tpkt, skip_cotp, find_mms_function_code

def extract_mms_control_code(packet) -> Optional[str]:
    """
    Extract MMS control code signature from a TCP packet.
    Uses shared functions from specification_mms.py.
    
    Args:
        packet: A Scapy packet that may contain an MMS message over TCP (ISO-on-TCP, port 102)
        
    Returns:
        A string representing the MMS control code, or None if not identifiable.
    """
    if not packet.haslayer(TCP):
        return None
    tcp = packet[TCP]
    if tcp.dport != 102 and tcp.sport != 102:
        return None
    if not packet.haslayer(Raw):
        return None

    raw = bytes(packet[Raw].load)
    if len(raw) < 7:
        return None

    # TPKT
    off = skip_tpkt(raw, 0)
    if off is None:
        return None

    # COTP
    res = skip_cotp(raw, off)
    if res is None:
        return None
    off, eot_flag = res

    # Find MMS function code
    tag_off = find_mms_function_code(raw, off)
    
    if tag_off is None:
        return f"{eot_flag}:None"
    else:
        control_code = f"{eot_flag}:0x{raw[tag_off]:02X}"
        return control_code