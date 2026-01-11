"""
Real-time analysis routes
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from flask import Blueprint, render_template, jsonify, request, Response, current_app
import queue

realtime_bp = Blueprint('realtime', __name__)

# Global state for real-time analysis
_realtime_state = {
    'pipeline': None,
    'running': False,
    'flows': [],
    'stats': {},
    'events_queue': queue.Queue(maxsize=1000),
    'lock': threading.Lock()
}


def get_pipeline():
    """Get or create the realtime pipeline"""
    from realtime import RealtimePipeline
    return _realtime_state.get('pipeline')


@realtime_bp.route('/')
def realtime_page():
    """Real-time analysis page"""
    # Get available interfaces
    try:
        from realtime.capture import PacketCapture
        interfaces = PacketCapture.list_interfaces()
        # Filter to main interfaces
        main_interfaces = []
        for iface in interfaces:
            name = iface.get('name', '')
            ips = iface.get('ips', [])
            real_ips = [ip for ip in ips
                        if not ip.startswith('169.254')
                        and not ip.startswith('fe80')
                        and ip not in ('127.0.0.1', '::1')]
            if real_ips and not any(x in name for x in ['Npcap', 'WFP', 'Filter', 'Loopback']):
                main_interfaces.append({
                    'name': name,
                    'ips': real_ips[:2]
                })
    except Exception as e:
        main_interfaces = []

    return render_template('realtime.html', interfaces=main_interfaces)


@realtime_bp.route('/start', methods=['POST'])
def start_capture():
    """Start real-time capture"""
    global _realtime_state

    if _realtime_state['running']:
        return jsonify({'status': 'error', 'message': 'Already running'})

    data = request.get_json() or {}
    interface = data.get('interface')
    threshold = float(data.get('threshold', 0.5))

    if not interface:
        return jsonify({'status': 'error', 'message': 'No interface specified'})

    try:
        from realtime import RealtimePipeline

        project_root = current_app.config['PROJECT_ROOT']
        model_path = project_root / 'training_artifacts' / 'best_model_XGB_regularized.joblib'
        preprocessor_path = project_root / 'artifacts' / 'preprocessor.joblib'
        schema_path = project_root / 'artifacts' / 'feature_schema.json'

        # Clear previous state
        with _realtime_state['lock']:
            _realtime_state['flows'] = []
            _realtime_state['stats'] = {}
            # Clear queue
            while not _realtime_state['events_queue'].empty():
                try:
                    _realtime_state['events_queue'].get_nowait()
                except:
                    pass

        def on_flow(result):
            """Callback for each analyzed flow"""
            flow_data = {
                'timestamp': result.timestamp,
                'src_ip': result.src_ip,
                'src_port': result.src_port,
                'dst_ip': result.dst_ip,
                'dst_port': result.dst_port,
                'protocol': result.protocol,
                'packets': result.total_packets,
                'bytes': result.total_bytes,
                'is_attack': result.is_attack,
                'confidence': result.confidence,
                'class_name': result.class_name
            }

            with _realtime_state['lock']:
                _realtime_state['flows'].append(flow_data)
                # Keep last 100 flows
                if len(_realtime_state['flows']) > 100:
                    _realtime_state['flows'] = _realtime_state['flows'][-100:]

            # Push to SSE queue
            try:
                _realtime_state['events_queue'].put_nowait({
                    'type': 'flow',
                    'data': flow_data
                })
            except queue.Full:
                pass

        def on_attack(result):
            """Callback for detected attacks"""
            attack_data = {
                'timestamp': result.timestamp,
                'src_ip': result.src_ip,
                'src_port': result.src_port,
                'dst_ip': result.dst_ip,
                'dst_port': result.dst_port,
                'confidence': result.confidence
            }
            try:
                _realtime_state['events_queue'].put_nowait({
                    'type': 'attack',
                    'data': attack_data
                })
            except queue.Full:
                pass

        pipeline = RealtimePipeline(
            interface=interface,
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path),
            feature_schema_path=str(schema_path),
            threshold=threshold,
            on_flow_analyzed=on_flow,
            on_attack_detected=on_attack
        )

        pipeline.start()
        time.sleep(1)

        if not pipeline.is_running():
            return jsonify({'status': 'error', 'message': 'Failed to start capture'})

        _realtime_state['pipeline'] = pipeline
        _realtime_state['running'] = True

        return jsonify({'status': 'success', 'message': 'Capture started'})

    except Exception as e:
        import traceback
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()})


@realtime_bp.route('/stop', methods=['POST'])
def stop_capture():
    """Stop real-time capture"""
    global _realtime_state

    if not _realtime_state['running']:
        return jsonify({'status': 'error', 'message': 'Not running'})

    try:
        if _realtime_state['pipeline']:
            _realtime_state['pipeline'].stop()
            _realtime_state['pipeline'] = None

        _realtime_state['running'] = False

        return jsonify({'status': 'success', 'message': 'Capture stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@realtime_bp.route('/stats')
def get_stats():
    """Get current statistics"""
    global _realtime_state

    if _realtime_state['pipeline'] and _realtime_state['running']:
        try:
            summary = _realtime_state['pipeline'].get_summary()
            return jsonify({
                'running': True,
                'packets': summary.get('total_packets', 0),
                'flows': summary.get('total_flows_analyzed', 0),
                'attacks': summary.get('total_attacks', 0),
                'packets_per_sec': summary.get('packets_per_second', 0),
                'active_flows': summary.get('active_flows', 0),
                'attack_rate': summary.get('recent_attack_rate', 0) * 100
            })
        except:
            pass

    return jsonify({
        'running': False,
        'packets': 0,
        'flows': 0,
        'attacks': 0,
        'packets_per_sec': 0,
        'active_flows': 0,
        'attack_rate': 0
    })


@realtime_bp.route('/flows')
def get_flows():
    """Get recent flows"""
    global _realtime_state

    with _realtime_state['lock']:
        flows = list(_realtime_state['flows'])

    return jsonify(flows[-50:])  # Last 50 flows


@realtime_bp.route('/events')
def events():
    """Server-Sent Events stream for real-time updates"""

    def generate():
        while True:
            try:
                # Get stats every second
                if _realtime_state['running'] and _realtime_state['pipeline']:
                    try:
                        summary = _realtime_state['pipeline'].get_summary()
                        stats_data = {
                            'type': 'stats',
                            'data': {
                                'running': True,
                                'packets': summary.get('total_packets', 0),
                                'flows': summary.get('total_flows_analyzed', 0),
                                'attacks': summary.get('total_attacks', 0),
                                'packets_per_sec': round(summary.get('packets_per_second', 0), 1),
                                'active_flows': summary.get('active_flows', 0),
                                'attack_rate': round(summary.get('recent_attack_rate', 0) * 100, 2)
                            }
                        }
                        yield f"data: {json.dumps(stats_data)}\n\n"
                    except:
                        pass

                # Check for flow events
                try:
                    event = _realtime_state['events_queue'].get(timeout=1)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    # Send heartbeat
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

            except GeneratorExit:
                break
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                time.sleep(1)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
