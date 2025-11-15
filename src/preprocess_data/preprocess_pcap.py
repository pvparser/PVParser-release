from pathlib import Path
from scapy.all import PcapReader, wrpcap
from scapy.layers.inet import IP, TCP, UDP
import hashlib
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basis.ics_basis import *


def split_pcap_by_packet_count(pcap_file_path, packets_per_file=1000000, output_folder=None):
    """
    Splits a large PCAP file into multiple smaller PCAP files based on packet count.

    Parameters
    ----------
        file_path (str): Path to the input PCAP file.
        packets_per_file (int): Number of packets per output file. Default is 1000.

    Returns
    -------
        None. Writes split PCAP files to a subdirectory named after the original file.
    """
    print(f"\nSplitting {pcap_file_path} into {packets_per_file} packets per file")
    if not output_folder:
        print("Output folder is not provided, please provide a valid output folder path")
        return
    
    os.makedirs(output_folder, exist_ok=True)

    reader = PcapReader(str(pcap_file_path))
    split_packets = []
    file_index = 0
    packet_count = 0

    pcap_file_name = pcap_file_path.split("/")[-1].replace(".pcap", "")
    for pkt in reader:
        split_packets.append(pkt)
        packet_count += 1

        if packet_count >= packets_per_file:
            output_file = os.path.join(output_folder, f"{pcap_file_name}_part{file_index}.pcap")
            wrpcap(output_file, split_packets)
            print(f"  Wrote {len(split_packets)} packets to {output_file}")
            split_packets = []
            packet_count = 0
            file_index += 1

    if split_packets:
        output_file = os.path.join(output_folder, f"{pcap_file_name}_part{file_index}.pcap")
        wrpcap(output_file, split_packets)
        print(f"  Wrote {len(split_packets)} packets to {output_file}")

    reader.close()


def extract_packets_with_filters(file_path, filters, output_folder):
    """
    Apply a sequence of packet list filters to the input pcap file,
    and write the filtered packets to a new pcap file with '_filtered' suffix.

    Parameters
    ----------
    file_path : str or Path
        Path to the input pcap file.
    filters : list of functions
        Each function takes a list of packets and returns a filtered list.

    Returns
    -------
    None
    """
    file_path = Path(file_path)

    # Read all packets from the pcap file
    with PcapReader(str(file_path)) as reader:
        packets = list(reader)

    # Apply each filter to the packet list
    for f in filters:
        packets = f(packets)

    # Generate output file path with '_filtered' suffix
    output_file = os.path.join(output_folder, file_path.stem + "_filtered" + file_path.suffix)

    # Write filtered packets to the new pcap file
    wrpcap(str(output_file), packets)
    print(f"  Wrote {len(packets)} packets to {output_file}")


def filter_retransmission(all_packets):
    """
    Filter out retransmission packets and their corresponding response packets, if response packets are also duplicate.

    Parameters
    ----------
    packets

    Returns
    -------
    filtered_packets: packets without retransmission
    """
    filtered_packets = []
    seen_hashes1 = set()
    seen_hashes2 = set()# Track packet hashes
    for pkt in all_packets:
        pkt_hash1, pkt_hash2 = packet_hash(pkt)
        if pkt_hash1 and pkt_hash1 in seen_hashes1:
            continue
                # print("Resubmission detected:", pkt.summary())
        elif pkt_hash2 and pkt_hash2 in seen_hashes2:
            continue
        else:
            if pkt_hash1: seen_hashes1.add(pkt_hash1)
            if pkt_hash2: seen_hashes2.add(pkt_hash2)
            filtered_packets.append(pkt)
    
    print(f"filter_retransmission ------> The number of all packets: {len(all_packets)}")
    print(f"filter_retransmission ------> The number of filtered packets: {len(filtered_packets)}")
    return filtered_packets


def packet_hash(packet):
    """
    Create a hash based on key header fields to uniquely identify packet data.

    Parameters
    ----------
    packet:

    Returns
    -------
    : hash value of a packet
    """
    if packet.haslayer(IP) and (packet.haslayer(TCP) or packet.haslayer(UDP)):
        # Select header fields to hash (IP + TCP headers, including sequence and ack numbers)
        src = packet[IP].src
        dst = packet[IP].dst
        sport = packet[TCP].sport
        dport = packet[TCP].dport
        proto = packet[IP].proto
        seq = packet[TCP].seq  # the response packet to the retransmission packet will have a different sequence number, when some new packets are sent before the duplicate response packets
        ack = packet[TCP].ack  # But its ack number still keeps consistent with the retransmission packet.
        flags = packet[TCP].flags.value
        payload = bytes(packet[TCP].payload)

        # Hash considering flags
        hash1 = hashlib.md5(f"{src}-{dst}-{sport}-{dport}-{proto}-{ack}-{flags}-{payload}".encode()).hexdigest()
        # Hash considering seq and the packets with payload, because the seq of packets like ACK and FIN ACK won't change. Therefore, hash2 is not applicable
        hash2 = hashlib.md5(f"{src}-{dst}-{sport}-{dport}-{proto}-{ack}-{seq}-{payload}".encode()).hexdigest() if payload else None
        return hash1, hash2
    return None, None


def filter_ics_protocol(all_packets, protocol):
    """
    Filter non-TCP and non_ICS packets.

    Parameters
    ----------
    all_packets: all packets from the pcap file

    Returns
    -------
    protocol_filtered_packets: filtered packets from the pcap file
    """
    # Filter out packets of non-ICS and non-TCP packets
    protocol_filtered_packets = [pkt for pkt in all_packets if TCP in pkt and (is_ics_port_by_protocol(pkt[TCP].sport, protocol) or is_ics_port_by_protocol(pkt[TCP].dport, protocol))]

    print(f"filter_ics_protocol ({protocol.value}) ------> The number of all packets: {len(all_packets)}")
    print(f"filter_ics_protocol ({protocol.value}) ------> The number of filtered packets: {len(protocol_filtered_packets)}")
    return protocol_filtered_packets


def filter_handshake(all_packets):
    """
    Filter out handshake and rest packets.

    Parameters
    ----------
    all_packets: all packets from the pcap file

    Returns
    -------
    handshake_filtered_packets: filtered packets from the pcap file
    """

    # Filter out handshake packets (SYN, SYN-ACK, ACK after SYN-ACK, and Rest)
    handshake_filtered_packets = []
    handshake_tracker = {}  # Dictionary to track SYN and SYN-ACK packets by TCP connection (based on IP and ports)
    ip_set = set()
    for pkt in all_packets:
        if IP in pkt and TCP in pkt:
            ip_src = pkt[IP].src
            ip_dst = pkt[IP].dst
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport

            # Create a connection identifier
            connection_id = (ip_src, src_port, ip_dst, dst_port)
            # Check for SYN packet
            if pkt[TCP].flags == "S":  # SYN
                handshake_tracker[connection_id] = "SYN_SENT"
            # Check for SYN-ACK packet in response to SYN
            elif pkt[TCP].flags == "SA":  # SYN-ACK, including retransmitted SYN-ACK
                handshake_tracker[(ip_dst, dst_port, ip_src, src_port)] = "SYN-ACK_SENT"
            # Check for final ACK packet that completes the handshake
            elif pkt[TCP].flags == "A" and handshake_tracker.get(connection_id) == "SYN-ACK_SENT":
                # Reset the tracker for this connection if needed
                del handshake_tracker[connection_id]
            # Check for RST packets
            elif pkt[TCP].flags == "R":  # Reset
                continue
            # Check for RST, ACK packets
            elif pkt[TCP].flags == "RA":  # Reset
                continue
            else:
                handshake_filtered_packets.append(pkt)
                ip_set.add(pkt[IP].src)
                ip_set.add(pkt[IP].dst)
    print(f"filter_handshake ------> The number of all packets: {len(all_packets)}")
    print(f"filter_handshake ------> The number of filtered packets: {len(handshake_filtered_packets)}")
    return handshake_filtered_packets
