# -- Coding: utf-8 --
# @Version: 1.0.0
# @Time: 2024/11/22 16:41

from enum import Enum

class ICSProtocol(Enum):
    """ICS Protocol enumeration to avoid typos and provide type safety"""
    MMS = "mms"
    MODBUS = "modbus"
    ENIP = "enip"

ics_ports = {
    102: "mms",  # Both S7 and MMS can use port 102, here we use MMS as the ICS protocol
    502: "modbus/tcp",
    503: "modbus/tcp",
    44818: "ethernet/ip",
}

ics_protocol_ports = {
    ICSProtocol.MODBUS.value: [502, 503],
    ICSProtocol.ENIP.value: [44818],
    ICSProtocol.MMS.value: [102]  # MMS typically uses port 102, same as S7, here we use MMS as the ICS protocol
}


def is_ics_port_by_protocol(port, protocol):
    """
    Determine if the port is an ICS port

    Parameters
    ----------
    port : int
        Port number to check
    protocol : str or ICSProtocol
        Protocol name (string) or ICSProtocol enum value

    Returns
    -------
    bool
        True: an ICS port, False: otherwise
    """
    # Handle both string and enum inputs
    if isinstance(protocol, ICSProtocol):
        protocol = protocol.value
    
    my_ports = ics_protocol_ports.get(protocol) # Sometimes, ICS ports may be used as source ports, like 44818 of enip
    if my_ports is None:
        return False
    if port in my_ports:
        return True
    return False


def get_supported_protocols():
    """
    Get list of supported ICS protocols
    
    Returns
    -------
    list
        List of supported protocol names
    """
    return [protocol.value for protocol in ICSProtocol]


def get_protocol_ports(protocol):
    """
    Get ports for a specific protocol
    
    Parameters
    ----------
    protocol : str or ICSProtocol
        Protocol name or enum
        
    Returns
    -------
    list or None
        List of ports for the protocol, or None if not found
    """
    if isinstance(protocol, ICSProtocol):
        protocol = protocol.value
    return ics_protocol_ports.get(protocol)


# Convenience constants for common protocols
MMS = ICSProtocol.MMS.value
MODBUS = ICSProtocol.MODBUS.value
ENIP = ICSProtocol.ENIP.value
