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
    'lock': threading.Lock(),
    'project_root': None
}


def get_interfaces():
    """Get list of network interfaces"""
    try:
        import sys
        from pathlib import Path

        # Add project root to path if needed
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

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

            # Filter out virtual interfaces
            skip_keywords = ['Npcap', 'WFP', 'Filter', 'Loopback', 'Pseudo',
                             'Tunneling', 'isatap', 'Teredo', '6to4']
            should_skip = any(kw.lower() in name.lower() for kw in skip_keywords)

            if real_ips and not should_skip:
                main_interfaces.append({
                    'name': name,
                    'description': iface.get('description', ''),
                    'ips': real_ips[:2]
                })

        return main_interfaces
    except Exception as e:
        print(f"Error getting interfaces: {e}")
        return []


@realtime_bp.route('/')
def realtime_page():
    """Real-time analysis page"""
    interfaces = get_interfaces()
    return render_template('realtime.html', interfaces=interfaces)


@realtime_bp.route('/interfaces')
def list_interfaces():
    """API endpoint to get interfaces"""
    interfaces = get_interfaces()
    return jsonify(interfaces)


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

    # Store project root before starting
    project_root = current_app.config['PROJECT_ROOT']
    _realtime_state['project_root'] = project_root

    try:
        import sys
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from realtime import RealtimePipeline

        model_path = project_root / 'training_artifacts' / 'best_model_XGB_regularized.joblib'
        preprocessor_path = project_root / 'artifacts' / 'preprocessor.joblib'
        schema_path = project_root / 'artifacts' / 'feature_schema.json'

        # Verify files exist
        if not model_path.exists():
            return jsonify({'status': 'error', 'message': f'Model not found: {model_path}'})
        if not preprocessor_path.exists():
            return jsonify({'status': 'error', 'message': f'Preprocessor not found: {preprocessor_path}'})
        if not schema_path.exists():
            return jsonify({'status': 'error', 'message': f'Schema not found: {schema_path}'})

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
        time.sleep(1.5)

        if not pipeline.is_running():
            return jsonify({'status': 'error', 'message': 'Failed to start capture. Make sure you have admin rights.'})

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
                'packets_per_sec': round(summary.get('packets_per_second', 0), 1),
                'active_flows': summary.get('active_flows', 0),
                'attack_rate': round(summary.get('recent_attack_rate', 0) * 100, 2)
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

    return jsonify(flows[-50:])


@realtime_bp.route('/events')
def events():
    """Server-Sent Events stream for real-time updates"""

    def generate():
        last_stats_time = 0

        while True:
            try:
                current_time = time.time()

                # Send stats every second
                if current_time - last_stats_time >= 1:
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
                        except Exception as e:
                            pass
                    else:
                        # Send offline status
                        yield f"data: {json.dumps({'type': 'stats', 'data': {'running': False}})}\n\n"

                    last_stats_time = current_time

                # Check for flow events (non-blocking)
                try:
                    event = _realtime_state['events_queue'].get(timeout=0.5)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    pass

            except GeneratorExit:
                break
            except Exception as e:
                time.sleep(1)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )
