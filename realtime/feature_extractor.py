"""
Feature extraction from network flows
Builds features compatible with the CICIDS2017 dataset
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Feature configuration"""
    # List of CICIDS2017 features that we can extract
    FEATURE_NAMES = [
        'Destination Port',
        'Flow Duration',
        'Total Fwd Packets',
        'Total Backward Packets',
        'Total Length of Fwd Packets',
        'Total Length of Bwd Packets',
        'Fwd Packet Length Max',
        'Fwd Packet Length Min',
        'Fwd Packet Length Mean',
        'Fwd Packet Length Std',
        'Bwd Packet Length Max',
        'Bwd Packet Length Min',
        'Bwd Packet Length Mean',
        'Bwd Packet Length Std',
        'Flow Bytes/s',
        'Flow Packets/s',
        'Flow IAT Mean',
        'Flow IAT Std',
        'Flow IAT Max',
        'Flow IAT Min',
        'Fwd IAT Total',
        'Fwd IAT Mean',
        'Fwd IAT Std',
        'Fwd IAT Max',
        'Fwd IAT Min',
        'Bwd IAT Total',
        'Bwd IAT Mean',
        'Bwd IAT Std',
        'Bwd IAT Max',
        'Bwd IAT Min',
        'Fwd PSH Flags',
        'Bwd PSH Flags',
        'Fwd URG Flags',
        'Bwd URG Flags',
        'Fwd Header Length',
        'Bwd Header Length',
        'Fwd Packets/s',
        'Bwd Packets/s',
        'Min Packet Length',
        'Max Packet Length',
        'Packet Length Mean',
        'Packet Length Std',
        'Packet Length Variance',
        'FIN Flag Count',
        'SYN Flag Count',
        'RST Flag Count',
        'PSH Flag Count',
        'ACK Flag Count',
        'URG Flag Count',
        'CWE Flag Count',
        'ECE Flag Count',
        'Down/Up Ratio',
        'Average Packet Size',
        'Avg Fwd Segment Size',
        'Avg Bwd Segment Size',
        'Fwd Avg Bytes/Bulk',
        'Fwd Avg Packets/Bulk',
        'Fwd Avg Bulk Rate',
        'Bwd Avg Bytes/Bulk',
        'Bwd Avg Packets/Bulk',
        'Bwd Avg Bulk Rate',
        'Subflow Fwd Packets',
        'Subflow Fwd Bytes',
        'Subflow Bwd Packets',
        'Subflow Bwd Bytes',
        'Init_Win_bytes_forward',
        'Init_Win_bytes_backward',
        'act_data_pkt_fwd',
        'min_seg_size_forward',
        'Active Mean',
        'Active Std',
        'Active Max',
        'Active Min',
        'Idle Mean',
        'Idle Std',
        'Idle Max',
        'Idle Min',
    ]


class FeatureExtractor:
    """
    Extracts features from aggregated flows
    Output format is compatible with CICIDS2017
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Args:
            feature_names: List of feature names to extract
                           (None = all available)
        """
        self.feature_names = feature_names or FeatureConfig.FEATURE_NAMES
        self._eps = 1e-10  # Avoid division by zero

    def _safe_stat(self, values: List[float], stat: str) -> float:
        """Safely computes a statistic"""
        if not values:
            return 0.0

        arr = np.array(values, dtype=np.float64)

        if stat == 'mean':
            return float(np.mean(arr))
        elif stat == 'std':
            return float(np.std(arr)) if len(arr) > 1 else 0.0
        elif stat == 'max':
            return float(np.max(arr))
        elif stat == 'min':
            return float(np.min(arr))
        elif stat == 'sum':
            return float(np.sum(arr))
        elif stat == 'var':
            return float(np.var(arr)) if len(arr) > 1 else 0.0
        else:
            return 0.0

    def extract(self, flow_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extracts features from flow data

        Args:
            flow_data: Flow data dictionary from FlowAggregator

        Returns:
            Feature dictionary
        """
        features = {}

        # Base data
        duration = max(flow_data.get('duration', 0), self._eps)
        fwd_packets = flow_data.get('total_fwd_packets', 0)
        bwd_packets = flow_data.get('total_bwd_packets', 0)
        total_packets = fwd_packets + bwd_packets

        fwd_bytes = flow_data.get('total_fwd_bytes', 0)
        bwd_bytes = flow_data.get('total_bwd_bytes', 0)
        total_bytes = fwd_bytes + bwd_bytes

        fwd_lengths = flow_data.get('fwd_packet_lengths', [])
        bwd_lengths = flow_data.get('bwd_packet_lengths', [])
        all_lengths = fwd_lengths + bwd_lengths

        fwd_iat = flow_data.get('fwd_iat', [])
        bwd_iat = flow_data.get('bwd_iat', [])
        all_iat = fwd_iat + bwd_iat

        # Feature extraction
        features['Destination Port'] = float(flow_data.get('dst_port', 0))
        features['Flow Duration'] = duration * 1e6  # Microseconds

        # Packet counts
        features['Total Fwd Packets'] = float(fwd_packets)
        features['Total Backward Packets'] = float(bwd_packets)

        # Packet lengths
        features['Total Length of Fwd Packets'] = float(fwd_bytes)
        features['Total Length of Bwd Packets'] = float(bwd_bytes)

        features['Fwd Packet Length Max'] = self._safe_stat(fwd_lengths, 'max')
        features['Fwd Packet Length Min'] = self._safe_stat(fwd_lengths, 'min')
        features['Fwd Packet Length Mean'] = self._safe_stat(fwd_lengths, 'mean')
        features['Fwd Packet Length Std'] = self._safe_stat(fwd_lengths, 'std')

        features['Bwd Packet Length Max'] = self._safe_stat(bwd_lengths, 'max')
        features['Bwd Packet Length Min'] = self._safe_stat(bwd_lengths, 'min')
        features['Bwd Packet Length Mean'] = self._safe_stat(bwd_lengths, 'mean')
        features['Bwd Packet Length Std'] = self._safe_stat(bwd_lengths, 'std')

        # Flow rates
        features['Flow Bytes/s'] = total_bytes / duration
        features['Flow Packets/s'] = total_packets / duration

        # IAT (Inter-Arrival Time)
        features['Flow IAT Mean'] = self._safe_stat(all_iat, 'mean') * 1e6
        features['Flow IAT Std'] = self._safe_stat(all_iat, 'std') * 1e6
        features['Flow IAT Max'] = self._safe_stat(all_iat, 'max') * 1e6
        features['Flow IAT Min'] = self._safe_stat(all_iat, 'min') * 1e6

        features['Fwd IAT Total'] = self._safe_stat(fwd_iat, 'sum') * 1e6
        features['Fwd IAT Mean'] = self._safe_stat(fwd_iat, 'mean') * 1e6
        features['Fwd IAT Std'] = self._safe_stat(fwd_iat, 'std') * 1e6
        features['Fwd IAT Max'] = self._safe_stat(fwd_iat, 'max') * 1e6
        features['Fwd IAT Min'] = self._safe_stat(fwd_iat, 'min') * 1e6

        features['Bwd IAT Total'] = self._safe_stat(bwd_iat, 'sum') * 1e6
        features['Bwd IAT Mean'] = self._safe_stat(bwd_iat, 'mean') * 1e6
        features['Bwd IAT Std'] = self._safe_stat(bwd_iat, 'std') * 1e6
        features['Bwd IAT Max'] = self._safe_stat(bwd_iat, 'max') * 1e6
        features['Bwd IAT Min'] = self._safe_stat(bwd_iat, 'min') * 1e6

        # TCP flags
        features['Fwd PSH Flags'] = float(flow_data.get('fwd_psh_flags', 0))
        features['Bwd PSH Flags'] = float(flow_data.get('bwd_psh_flags', 0))
        features['Fwd URG Flags'] = float(flow_data.get('fwd_urg_flags', 0))
        features['Bwd URG Flags'] = float(flow_data.get('bwd_urg_flags', 0))

        # Header lengths
        features['Fwd Header Length'] = float(flow_data.get('fwd_header_length', 0))
        features['Bwd Header Length'] = float(flow_data.get('bwd_header_length', 0))

        # Packets per second
        features['Fwd Packets/s'] = fwd_packets / duration
        features['Bwd Packets/s'] = bwd_packets / duration

        # Packet length statistics
        features['Min Packet Length'] = self._safe_stat(all_lengths, 'min')
        features['Max Packet Length'] = self._safe_stat(all_lengths, 'max')
        features['Packet Length Mean'] = self._safe_stat(all_lengths, 'mean')
        features['Packet Length Std'] = self._safe_stat(all_lengths, 'std')
        features['Packet Length Variance'] = self._safe_stat(all_lengths, 'var')

        # Flag counts
        features['FIN Flag Count'] = float(flow_data.get('fin_count', 0))
        features['SYN Flag Count'] = float(flow_data.get('syn_count', 0))
        features['RST Flag Count'] = float(flow_data.get('rst_count', 0))
        features['PSH Flag Count'] = float(flow_data.get('psh_count', 0))
        features['ACK Flag Count'] = float(flow_data.get('ack_count', 0))
        features['URG Flag Count'] = float(flow_data.get('urg_count', 0))
        features['CWE Flag Count'] = float(flow_data.get('cwr_count', 0))
        features['ECE Flag Count'] = float(flow_data.get('ece_count', 0))

        # Ratios
        if fwd_packets > 0:
            features['Down/Up Ratio'] = bwd_packets / fwd_packets
        else:
            features['Down/Up Ratio'] = 0.0

        # Average sizes
        if total_packets > 0:
            features['Average Packet Size'] = total_bytes / total_packets
        else:
            features['Average Packet Size'] = 0.0

        if fwd_packets > 0:
            features['Avg Fwd Segment Size'] = fwd_bytes / fwd_packets
        else:
            features['Avg Fwd Segment Size'] = 0.0

        if bwd_packets > 0:
            features['Avg Bwd Segment Size'] = bwd_bytes / bwd_packets
        else:
            features['Avg Bwd Segment Size'] = 0.0

        # Bulk averages (simplified version)
        features['Fwd Avg Bytes/Bulk'] = 0.0
        features['Fwd Avg Packets/Bulk'] = 0.0
        features['Fwd Avg Bulk Rate'] = 0.0
        features['Bwd Avg Bytes/Bulk'] = 0.0
        features['Bwd Avg Packets/Bulk'] = 0.0
        features['Bwd Avg Bulk Rate'] = 0.0

        # Subflow (for a single flow, this equals the main values)
        features['Subflow Fwd Packets'] = float(fwd_packets)
        features['Subflow Fwd Bytes'] = float(fwd_bytes)
        features['Subflow Bwd Packets'] = float(bwd_packets)
        features['Subflow Bwd Bytes'] = float(bwd_bytes)

        # Init window bytes (requires deeper TCP parsing)
        features['Init_Win_bytes_forward'] = 0.0
        features['Init_Win_bytes_backward'] = 0.0

        # Active data packets
        features['act_data_pkt_fwd'] = float(flow_data.get('fwd_payload_bytes', 0) > 0)
        features['min_seg_size_forward'] = self._safe_stat(fwd_lengths, 'min')

        # Active/idle times (simplified version)
        features['Active Mean'] = 0.0
        features['Active Std'] = 0.0
        features['Active Max'] = 0.0
        features['Active Min'] = 0.0
        features['Idle Mean'] = 0.0
        features['Idle Std'] = 0.0
        features['Idle Max'] = 0.0
        features['Idle Min'] = 0.0

        return features

    def extract_array(self, flow_data: Dict[str, Any]) -> np.ndarray:
        """
        Extracts features as a numpy array

        Args:
            flow_data: Flow data

        Returns:
            Feature array
        """
        features = self.extract(flow_data)
        return np.array([features.get(name, 0.0) for name in self.feature_names],
                       dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """Returns the list of feature names"""
        return self.feature_names.copy()

    def extract_batch(self, flows: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extracts features for a batch of flows

        Args:
            flows: List of flows

        Returns:
            2D feature array [n_flows, n_features]
        """
        if not flows:
            return np.array([]).reshape(0, len(self.feature_names))

        return np.vstack([self.extract_array(flow) for flow in flows])

    def get_feature_count(self) -> int:
        """Returns the number of features"""
        return len(self.feature_names)


# Testing
if __name__ == "__main__":
    # Test flow data
    test_flow = {
        'src_ip': '192.168.1.100',
        'dst_ip': '8.8.8.8',
        'src_port': 54321,
        'dst_port': 443,
        'protocol': 6,
        'duration': 1.5,
        'total_fwd_packets': 10,
        'total_bwd_packets': 8,
        'total_fwd_bytes': 1500,
        'total_bwd_bytes': 12000,
        'fwd_packet_lengths': [150, 150, 150, 150, 150, 150, 150, 150, 150, 150],
        'bwd_packet_lengths': [1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500],
        'fwd_iat': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'bwd_iat': [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
        'fwd_psh_flags': 5,
        'bwd_psh_flags': 3,
        'syn_count': 1,
        'ack_count': 18,
        'fin_count': 2,
    }

    extractor = FeatureExtractor()

    print("Feature Extractor Test")
    print("=" * 50)
    print(f"Number of features: {extractor.get_feature_count()}")
    print()

    # Extract features
    features_dict = extractor.extract(test_flow)
    features_array = extractor.extract_array(test_flow)

    print("Sample features (dict):")
    for name in list(features_dict.keys())[:10]:
        print(f"  {name}: {features_dict[name]:.4f}")
    print("  ...")

    print()
    print(f"Features array shape: {features_array.shape}")
    print(f"Features array dtype: {features_array.dtype}")
    print(f"First 5 values: {features_array[:5]}")
