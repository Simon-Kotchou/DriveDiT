"""
Inference server implementations for production deployment.
Includes HTTP REST API and WebSocket streaming interfaces.
"""

import asyncio
import json
import time
import base64
import io
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import numpy as np
import torch
from PIL import Image

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: pip install fastapi uvicorn")

from .pipeline import InferencePipeline, RealtimeInferencePipeline, PipelineConfig


class InferenceRequest(BaseModel):
    """Request model for inference API."""
    context_frames: List[str]  # Base64 encoded images
    controls: Optional[List[List[float]]] = None
    num_frames: int = 16
    temperature: float = 0.8
    return_intermediates: bool = False


class InferenceResponse(BaseModel):
    """Response model for inference API."""
    generated_frames: List[str]  # Base64 encoded images
    context_frames: List[str]
    metadata: Dict[str, Any]
    performance: Dict[str, float]
    success: bool = True
    error: Optional[str] = None


class StreamingMessage(BaseModel):
    """Message model for WebSocket streaming."""
    type: str  # 'frame', 'control', 'result', 'error'
    data: Union[str, Dict[str, Any]]
    timestamp: float


class InferenceServer:
    """Base inference server with model management."""
    
    def __init__(
        self,
        model_path: str,
        vae_path: str,
        device: str = 'cuda',
        max_batch_size: int = 4,
        enable_optimization: bool = True
    ):
        """
        Initialize inference server.
        
        Args:
            model_path: Path to model checkpoint
            vae_path: Path to VAE checkpoint
            device: Device to run inference on
            max_batch_size: Maximum batch size
            enable_optimization: Whether to enable model optimizations
        """
        self.device = device
        self.max_batch_size = max_batch_size
        
        # Initialize pipeline
        config = PipelineConfig(
            model_path=model_path,
            vae_path=vae_path,
            device=device,
            batch_size=max_batch_size,
            use_torch_compile=enable_optimization,
            use_mixed_precision=True
        )
        
        self.pipeline = InferencePipeline(config)
        self.realtime_pipeline = RealtimeInferencePipeline(config)
        
        # Server state
        self.active_connections: List[WebSocket] = []
        self.is_running = False
        
        print(f"Inference server initialized on {device}")
    
    def _decode_image(self, base64_str: str) -> np.ndarray:
        """Decode base64 image to numpy array."""
        try:
            # Remove data URL prefix if present
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(base64_str)
            
            # Load image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Transpose to CHW format
            image_array = image_array.transpose(2, 0, 1)
            
            return image_array
            
        except Exception as e:
            raise ValueError(f"Failed to decode image: {e}")
    
    def _encode_image(self, image_array: np.ndarray) -> str:
        """Encode numpy array to base64 image."""
        try:
            # Ensure image is in [0, 1] range
            image_array = np.clip(image_array, 0, 1)
            
            # Convert to HWC format if needed
            if image_array.shape[0] == 3:  # CHW format
                image_array = image_array.transpose(1, 2, 0)
            
            # Convert to uint8
            image_uint8 = (image_array * 255).astype(np.uint8)
            
            # Convert to PIL Image
            image = Image.fromarray(image_uint8)
            
            # Encode to base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=90)
            buffer.seek(0)
            
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_str}"
            
        except Exception as e:
            raise ValueError(f"Failed to encode image: {e}")
    
    def process_inference_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process inference request."""
        try:
            # Decode context frames
            context_frames = []
            for frame_b64 in request.context_frames:
                frame_array = self._decode_image(frame_b64)
                context_frames.append(frame_array)
            
            context_tensor = torch.tensor(np.stack(context_frames)).unsqueeze(0)  # Add batch dim
            
            # Process controls if provided
            controls_tensor = None
            if request.controls:
                controls_tensor = torch.tensor(request.controls).unsqueeze(0)  # Add batch dim
            
            # Run inference
            result = self.pipeline.generate(
                context_frames=context_tensor,
                controls=controls_tensor,
                num_frames=request.num_frames,
                return_dict=True
            )
            
            # Encode generated frames
            generated_b64 = []
            for frame in result['generated_frames'][0]:  # Remove batch dim
                frame_b64 = self._encode_image(frame)
                generated_b64.append(frame_b64)
            
            # Encode context frames for response
            context_b64 = []
            for frame in result['context_frames'][0]:  # Remove batch dim
                frame_b64 = self._encode_image(frame)
                context_b64.append(frame_b64)
            
            return InferenceResponse(
                generated_frames=generated_b64,
                context_frames=context_b64,
                metadata=result['metadata'],
                performance=result['performance'],
                success=True
            )
            
        except Exception as e:
            return InferenceResponse(
                generated_frames=[],
                context_frames=[],
                metadata={},
                performance={},
                success=False,
                error=str(e)
            )


class HTTPServer(InferenceServer):
    """HTTP REST API server for inference."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI required for HTTP server")
        
        self.app = FastAPI(
            title="DriveDiT Inference API",
            description="Real-time driving video generation API",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "device": self.device,
                "timestamp": time.time()
            }
        
        @self.app.post("/generate", response_model=InferenceResponse)
        async def generate_video(request: InferenceRequest):
            """Generate video sequence."""
            return self.process_inference_request(request)
        
        @self.app.get("/info")
        async def server_info():
            """Get server information."""
            return {
                "device": self.device,
                "max_batch_size": self.max_batch_size,
                "model_loaded": True,
                "capabilities": {
                    "batch_inference": True,
                    "streaming": True,
                    "mixed_precision": True
                }
            }
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the HTTP server."""
        print(f"Starting HTTP server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


class WebSocketServer(InferenceServer):
    """WebSocket server for real-time streaming inference."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI required for WebSocket server")
        
        self.app = FastAPI(
            title="DriveDiT Streaming API",
            description="Real-time streaming video generation API",
            version="1.0.0"
        )
        
        self._setup_websocket_routes()
    
    def _setup_websocket_routes(self):
        """Setup WebSocket routes."""
        
        @self.app.websocket("/stream")
        async def websocket_endpoint(websocket: WebSocket):
            await self.connect(websocket)
            try:
                await self.handle_streaming_session(websocket)
            except WebSocketDisconnect:
                self.disconnect(websocket)
            except Exception as e:
                await self.send_error(websocket, str(e))
                self.disconnect(websocket)
        
        @self.app.get("/")
        async def root():
            return {"message": "DriveDiT WebSocket Server"}
    
    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Start streaming pipeline
        self.realtime_pipeline.start_streaming()
        
        await self.send_message(websocket, StreamingMessage(
            type="connected",
            data={"status": "connected", "server_time": time.time()},
            timestamp=time.time()
        ))
    
    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Stop streaming if no active connections
        if not self.active_connections:
            self.realtime_pipeline.stop_streaming()
    
    async def send_message(self, websocket: WebSocket, message: StreamingMessage):
        """Send message through WebSocket."""
        await websocket.send_text(message.json())
    
    async def send_error(self, websocket: WebSocket, error: str):
        """Send error message."""
        message = StreamingMessage(
            type="error",
            data={"error": error},
            timestamp=time.time()
        )
        await self.send_message(websocket, message)
    
    async def handle_streaming_session(self, websocket: WebSocket):
        """Handle streaming session."""
        frame_count = 0
        
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message["type"] == "frame":
                    # Process incoming frame
                    frame_data = message["data"]
                    
                    # Decode frame
                    frame_array = self._decode_image(frame_data["image"])
                    
                    # Get control if provided
                    control = None
                    if "control" in frame_data:
                        control = np.array(frame_data["control"], dtype=np.float32)
                    
                    # Process frame through pipeline
                    result_frame = self.realtime_pipeline.process_frame(frame_array, control)
                    
                    if result_frame is not None:
                        # Encode and send result
                        result_b64 = self._encode_image(result_frame)
                        
                        response = StreamingMessage(
                            type="result",
                            data={
                                "generated_frame": result_b64,
                                "frame_id": frame_count,
                                "processing_time_ms": message.get("processing_time", 0)
                            },
                            timestamp=time.time()
                        )
                        
                        await self.send_message(websocket, response)
                        frame_count += 1
                
                elif message["type"] == "control":
                    # Handle control-only messages
                    control_data = message["data"]
                    
                    # Send acknowledgment
                    response = StreamingMessage(
                        type="control_ack",
                        data={"received": True},
                        timestamp=time.time()
                    )
                    await self.send_message(websocket, response)
                
                elif message["type"] == "ping":
                    # Handle ping
                    response = StreamingMessage(
                        type="pong",
                        data={"server_time": time.time()},
                        timestamp=time.time()
                    )
                    await self.send_message(websocket, response)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                await self.send_error(websocket, str(e))
    
    def run(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the WebSocket server."""
        print(f"Starting WebSocket server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


class MultiModalServer(InferenceServer):
    """Combined HTTP and WebSocket server."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI required for multi-modal server")
        
        self.app = FastAPI(
            title="DriveDiT Multi-Modal API",
            description="Combined HTTP and WebSocket API for video generation",
            version="1.0.0"
        )
        
        # Add CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_all_routes()
    
    def _setup_all_routes(self):
        """Setup both HTTP and WebSocket routes."""
        
        # HTTP routes
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "device": self.device,
                "active_connections": len(self.active_connections),
                "timestamp": time.time()
            }
        
        @self.app.post("/generate", response_model=InferenceResponse)
        async def generate_video(request: InferenceRequest):
            return self.process_inference_request(request)
        
        # WebSocket route
        @self.app.websocket("/stream")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            self.realtime_pipeline.start_streaming()
            
            try:
                await self._handle_websocket_session(websocket)
            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
                if not self.active_connections:
                    self.realtime_pipeline.stop_streaming()
    
    async def _handle_websocket_session(self, websocket: WebSocket):
        """Handle WebSocket session (simplified)."""
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Echo back for now (implement full streaming logic)
                await websocket.send_text(json.dumps({
                    "type": "echo",
                    "data": message,
                    "timestamp": time.time()
                }))
                
            except WebSocketDisconnect:
                break
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the multi-modal server."""
        print(f"Starting multi-modal server on {host}:{port}")
        print(f"  - HTTP API: http://{host}:{port}/docs")
        print(f"  - WebSocket: ws://{host}:{port}/stream")
        uvicorn.run(self.app, host=host, port=port)


def create_server(
    server_type: str,
    model_path: str,
    vae_path: str,
    **kwargs
) -> InferenceServer:
    """
    Factory function to create inference servers.
    
    Args:
        server_type: Type of server ('http', 'websocket', 'multimodal')
        model_path: Path to model checkpoint
        vae_path: Path to VAE checkpoint
        **kwargs: Additional server configuration
        
    Returns:
        Configured inference server
    """
    if server_type == 'http':
        return HTTPServer(model_path, vae_path, **kwargs)
    elif server_type == 'websocket':
        return WebSocketServer(model_path, vae_path, **kwargs)
    elif server_type == 'multimodal':
        return MultiModalServer(model_path, vae_path, **kwargs)
    else:
        raise ValueError(f"Unknown server type: {server_type}")


# CLI interface for running servers
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DriveDiT Inference Server")
    parser.add_argument("--server-type", choices=['http', 'websocket', 'multimodal'], 
                       default='multimodal', help="Type of server to run")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--vae-path", required=True, help="Path to VAE checkpoint")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--device", default="cuda", help="Device to run inference on")
    parser.add_argument("--batch-size", type=int, default=4, help="Maximum batch size")
    parser.add_argument("--no-optimization", action="store_true", 
                       help="Disable model optimizations")
    
    args = parser.parse_args()
    
    # Create and run server
    server = create_server(
        server_type=args.server_type,
        model_path=args.model_path,
        vae_path=args.vae_path,
        device=args.device,
        max_batch_size=args.batch_size,
        enable_optimization=not args.no_optimization
    )
    
    server.run(host=args.host, port=args.port)