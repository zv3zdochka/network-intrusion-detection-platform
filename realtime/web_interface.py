"""
Base web interface for integration with Flask/FastAPI
Provides endpoints for monitoring and control
"""

from typing import Dict, Any, Optional, List
from dataclasses import asdict
from datetime import datetime
import json

# Optional Flask import
try:
    from flask import Blueprint, jsonify, request

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Optional FastAPI import
try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class WebInterface:
    """
    Base web interface class
    Can be used with Flask or FastAPI
    """

    def __init__(self, pipeline):
        """
        Args:
            pipeline: RealtimePipeline instance
        """
        self.pipeline = pipeline

    def get_status(self) -> Dict[str, Any]:
        """Returns the current pipeline status"""
        return {
            'status': 'running' if self.pipeline.is_running() else 'stopped',
            'summary': self.pipeline.get_summary(),
            'timestamp': datetime.now().isoformat()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Returns detailed statistics"""
        return self.pipeline.get_stats()

    def get_recent_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Returns recent analysis results"""
        results = self.pipeline.get_recent_results(limit)
        return [asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in results]

    def get_recent_attacks(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Returns recent detected attacks"""
        attacks = self.pipeline.get_recent_attacks(limit)
        return [asdict(a) if hasattr(a, '__dataclass_fields__') else a for a in attacks]

    def get_active_flows(self) -> List[Dict[str, Any]]:
        """Returns active flows"""
        return self.pipeline.aggregator.get_active_flows()

    def start_pipeline(self) -> Dict[str, str]:
        """Starts the pipeline"""
        if self.pipeline.is_running():
            return {'status': 'already_running'}
        self.pipeline.start()
        return {'status': 'started'}

    def stop_pipeline(self) -> Dict[str, str]:
        """Stops the pipeline"""
        if not self.pipeline.is_running():
            return {'status': 'already_stopped'}
        self.pipeline.stop()
        return {'status': 'stopped'}


def create_flask_blueprint(pipeline) -> 'Blueprint':
    """
    Creates a Flask Blueprint with endpoints

    Args:
        pipeline: RealtimePipeline instance

    Returns:
        Flask Blueprint
    """
    if not FLASK_AVAILABLE:
        raise RuntimeError("Flask is not installed. Install with: pip install flask")

    bp = Blueprint('realtime', __name__, url_prefix='/api/realtime')
    interface = WebInterface(pipeline)

    @bp.route('/status', methods=['GET'])
    def status():
        return jsonify(interface.get_status())

    @bp.route('/stats', methods=['GET'])
    def stats():
        return jsonify(interface.get_stats())

    @bp.route('/results', methods=['GET'])
    def results():
        limit = request.args.get('limit', 100, type=int)
        return jsonify(interface.get_recent_results(limit))

    @bp.route('/attacks', methods=['GET'])
    def attacks():
        limit = request.args.get('limit', 100, type=int)
        return jsonify(interface.get_recent_attacks(limit))

    @bp.route('/flows', methods=['GET'])
    def flows():
        return jsonify(interface.get_active_flows())

    @bp.route('/start', methods=['POST'])
    def start():
        return jsonify(interface.start_pipeline())

    @bp.route('/stop', methods=['POST'])
    def stop():
        return jsonify(interface.stop_pipeline())

    return bp


def create_fastapi_router(pipeline) -> 'APIRouter':
    """
    Creates a FastAPI Router with endpoints

    Args:
        pipeline: RealtimePipeline instance

    Returns:
        FastAPI APIRouter
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is not installed. Install with: pip install fastapi")

    router = APIRouter(prefix="/api/realtime", tags=["realtime"])
    interface = WebInterface(pipeline)

    @router.get("/status")
    async def status():
        return interface.get_status()

    @router.get("/stats")
    async def stats():
        return interface.get_stats()

    @router.get("/results")
    async def results(limit: int = 100):
        return interface.get_recent_results(limit)

    @router.get("/attacks")
    async def attacks(limit: int = 100):
        return interface.get_recent_attacks(limit)

    @router.get("/flows")
    async def flows():
        return interface.get_active_flows()

    @router.post("/start")
    async def start():
        return interface.start_pipeline()

    @router.post("/stop")
    async def stop():
        return interface.stop_pipeline()

    return router


# WebSocket support for real-time updates
class WebSocketHandler:
    """WebSocket connection handler"""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.clients: List[Any] = []

    def add_client(self, websocket):
        """Adds a WebSocket client"""
        self.clients.append(websocket)

    def remove_client(self, websocket):
        """Removes a WebSocket client"""
        if websocket in self.clients:
            self.clients.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Sends a message to all clients"""
        data = json.dumps(message)
        for client in self.clients[:]:
            try:
                await client.send(data)
            except Exception:
                self.remove_client(client)

    def on_attack_detected(self, result):
        """Callback for broadcasting an attack notification"""
        import asyncio

        message = {
            'type': 'attack_detected',
            'data': asdict(result) if hasattr(result, '__dataclass_fields__') else result,
            'timestamp': datetime.now().isoformat()
        }

        # Schedule on the event loop
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self.broadcast(message))
        except RuntimeError:
            pass
