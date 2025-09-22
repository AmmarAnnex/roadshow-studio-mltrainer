#!/usr/bin/env python3
"""
Roadshow Studio v11a3b Inference Server - CLEARERVOICE ENHANCED VERSION
Preprocesses with ClearerVoice MossFormer2 before BigVGAN
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import librosa
import uvicorn
import io
import warnings
import tempfile
import subprocess
import uuid
from datetime import datetime, timedelta
import asyncio
import torchaudio
import torchaudio.functional as F
warnings.filterwarnings("ignore")

# Add paths for BigVGAN and ClearerVoice
sys.path.insert(0, '/workspace/repos/BigVGAN-Python/BigVGAN')
sys.path.insert(0, '/workspace/training/bigvgan_u87_training')
sys.path.insert(0, '/workspace/repos/ClearerVoice-Studio-main')

# Import BigVGAN components
from bigvgan import BigVGAN
from meldataset import get_mel_spectrogram

# Import ClearerVoice - MossFormer2
try:
    # Try to import the model directly
    from clearvoice.models import mossformer2_se
    CLEARVOICE_AVAILABLE = True
    print("✅ ClearerVoice MossFormer2 imported successfully")
except ImportError as e:
    print(f"⚠️ ClearerVoice import failed: {e}")
    print("Attempting alternative import...")
    try:
        # Alternative import path based on your directory structure
        sys.path.append('/workspace/repos/ClearerVoice-Studio-main/clearvoice')
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mossformer2_se",
            "/workspace/repos/ClearerVoice-Studio-main/clearvoice/models/mossformer2_se.py"
        )
        mossformer2_se = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mossformer2_se)
        CLEARVOICE_AVAILABLE = True
        print("✅ ClearerVoice loaded via alternative method")
    except:
        CLEARVOICE_AVAILABLE = False
        print("❌ ClearerVoice not available - will process without denoising")

app = FastAPI(title="Roadshow Studio v11a3b + ClearerVoice API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directory for temporary audio files
TEMP_DIR = "/workspace/temp_audio_clearvoice"
os.makedirs(TEMP_DIR, exist_ok=True)

# Global model variables
model = None
device = None
model_config = None
clearvoice_model = None

def load_clearvoice_model():
    """Load ClearerVoice MossFormer2 model"""
    global clearvoice_model
    
    if not CLEARVOICE_AVAILABLE:
        print("ClearerVoice not available, skipping...")
        return None
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the MossFormer2 model checkpoint
        # Try to find the model file
        model_path = "/workspace/repos/ClearerVoice-Studio-main/clearvoice/models/mossformer2_se"
        
        if os.path.exists(model_path):
            print(f"Loading ClearerVoice from {model_path}")
            # Load the pretrained model
            clearvoice_model = torch.jit.load(model_path, map_location=device)
            clearvoice_model.eval()
            print("✅ ClearerVoice model loaded successfully")
        else:
            print(f"⚠️ ClearerVoice model not found at {model_path}")
            # Try alternative loading method
            # Load from checkpoint if available
            checkpoint_paths = [
                "/workspace/repos/ClearerVoice-Studio-main/clearvoice/checkpoints/mossformer2_se.pth",
                "/workspace/repos/ClearerVoice-Studio-main/checkpoints/mossformer2_se.pth",
            ]
            
            for ckpt_path in checkpoint_paths:
                if os.path.exists(ckpt_path):
                    print(f"Loading checkpoint from {ckpt_path}")
                    clearvoice_model = torch.load(ckpt_path, map_location=device)
                    clearvoice_model.eval()
                    print("✅ ClearerVoice loaded from checkpoint")
                    break
                    
        return clearvoice_model
    except Exception as e:
        print(f"Error loading ClearerVoice: {e}")
        return None

def apply_clearvoice_denoising(audio_np, sr):
    """Apply ClearerVoice denoising to audio"""
    if clearvoice_model is None:
        print("ClearerVoice not loaded, returning original audio")
        return audio_np, sr
    
    try:
        # Convert numpy to tensor
        audio_tensor = torch.from_numpy(audio_np).float()
        
        # Ensure mono
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.mean(dim=0)
        
        # ClearerVoice expects 16kHz input
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio_16k = resampler(audio_tensor)
        else:
            audio_16k = audio_tensor
        
        # Add batch dimension
        audio_16k = audio_16k.unsqueeze(0)
        
        # Apply denoising
        with torch.no_grad():
            device = next(clearvoice_model.parameters()).device
            audio_16k = audio_16k.to(device)
            enhanced_16k = clearvoice_model(audio_16k)
            
            # Remove batch dimension
            enhanced_16k = enhanced_16k.squeeze(0)
        
        # Resample to 24kHz for BigVGAN
        resampler_24k = torchaudio.transforms.Resample(16000, 24000)
        enhanced_24k = resampler_24k(enhanced_16k.cpu())
        
        # Convert back to numpy
        enhanced_np = enhanced_24k.numpy()
        
        print("✅ ClearerVoice denoising applied successfully")
        return enhanced_np, 24000
        
    except Exception as e:
        print(f"Error in ClearerVoice processing: {e}")
        print("Falling back to original audio")
        # If denoising fails, just resample to 24kHz if needed
        if sr != 24000:
            audio_tensor = torch.from_numpy(audio_np).float()
            resampler = torchaudio.transforms.Resample(sr, 24000)
            audio_24k = resampler(audio_tensor)
            return audio_24k.numpy(), 24000
        return audio_np, sr

def load_v11a3b_model():
    """Load the v11a3b extended model with epoch 346 checkpoint"""
    global model, device, model_config
    
    # Checkpoint path - EPOCH 346
    checkpoint_path = "/workspace/training/bigvgan_u87_training/checkpoints/bigvgan_u87_v11a3b_phase_c50/bigvgan_u87_v11a_original_250_clean_epoch_346.pth"
    
    print(f"Loading v11a3b checkpoint: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load BigVGAN base model
    print("Loading BigVGAN base model...")
    model = BigVGAN.from_pretrained(
        "nvidia/bigvgan_v2_24khz_100band_256x",
        use_cuda_kernel=False
    )
    model.remove_weight_norm()
    
    # Load your fine-tuned checkpoint
    print("Loading fine-tuned weights...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'generator_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['generator_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval().to(device)
    
    # Store config for mel spectrogram generation
    model_config = {
        'sample_rate': 24000,
        'n_mel_channels': 100,
        'n_fft': 1024,
        'hop_length': 256,
        'win_length': 1024,
        'fmin': 0,
        'fmax': 12000
    }
    
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'losses' in checkpoint:
        val_loss = checkpoint['losses'].get('g_total', 'N/A')
        print(f"Validation loss: {val_loss}")
    
    print("✅ v11a3b model loaded successfully!")
    return model

def convert_audio_format(audio_bytes, original_filename="audio"):
    """Convert any audio format to WAV using ffmpeg as fallback"""
    try:
        # First try librosa
        audio_np, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        return audio_np, sr
    except:
        print(f"Librosa failed, trying ffmpeg for {original_filename}")
        
    # Fallback to ffmpeg
    try:
        with tempfile.NamedTemporaryFile(suffix=Path(original_filename).suffix, delete=False) as temp_in:
            temp_in.write(audio_bytes)
            temp_in_path = temp_in.name
        
        temp_out_path = tempfile.mktemp(suffix='.wav')
        
        cmd = [
            'ffmpeg', '-i', temp_in_path,
            '-ac', '1',      # Mono
            '-f', 'wav',
            temp_out_path,
            '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg conversion failed: {result.stderr}")
        
        audio_np, sr = librosa.load(temp_out_path, sr=None, mono=True)
        
        os.unlink(temp_in_path)
        os.unlink(temp_out_path)
        
        return audio_np, sr
        
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}")
        raise HTTPException(status_code=400, detail=f"Could not process audio format")

async def cleanup_old_files():
    """Remove audio files older than 1 hour"""
    while True:
        try:
            now = datetime.now()
            for filename in os.listdir(TEMP_DIR):
                filepath = os.path.join(TEMP_DIR, filename)
                file_age = now - datetime.fromtimestamp(os.path.getmtime(filepath))
                if file_age > timedelta(hours=1):
                    os.remove(filepath)
                    print(f"Cleaned up old file: {filename}")
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        await asyncio.sleep(1800)

@app.on_event("startup")
async def startup():
    """Load models on server startup"""
    try:
        # Load ClearerVoice first
        load_clearvoice_model()
        # Then load BigVGAN
        load_v11a3b_model()
        # Start cleanup task
        asyncio.create_task(cleanup_old_files())
    except Exception as e:
        print(f"ERROR loading models: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model": "v11a3b_clearvoice",
        "checkpoint": "epoch_346",
        "val_loss": "9.02",
        "clearvoice": "enabled" if clearvoice_model is not None else "disabled",
        "device": str(device) if device else "not loaded",
        "model_loaded": model is not None
    }

@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    clarity: float = 0.5,
    warmth: float = 0.7,
    use_clearvoice: bool = True
):
    """
    Process audio through ClearerVoice + v11a3b model
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read uploaded audio
        audio_bytes = await file.read()
        
        # Convert to proper format
        print(f"Processing file: {file.filename}, size: {len(audio_bytes)} bytes")
        audio_np, sr = convert_audio_format(audio_bytes, file.filename)
        
        print(f"Audio loaded: shape={audio_np.shape}, sr={sr}")
        
        # Apply ClearerVoice denoising if enabled
        if use_clearvoice and clearvoice_model is not None:
            print("Applying ClearerVoice denoising...")
            audio_np, sr = apply_clearvoice_denoising(audio_np, sr)
            print(f"After ClearerVoice: shape={audio_np.shape}, sr={sr}")
        elif use_clearvoice:
            print("ClearerVoice requested but not available")
        
        # Ensure 24kHz for BigVGAN
        if sr != 24000:
            audio_tensor = torch.from_numpy(audio_np).float()
            resampler = torchaudio.transforms.Resample(sr, 24000)
            audio_np = resampler(audio_tensor).numpy()
            sr = 24000
        
        # Normalize input
        if np.max(np.abs(audio_np)) > 0:
            audio_np = audio_np / np.max(np.abs(audio_np)) * 0.95
        
        # Process through BigVGAN model
        with torch.no_grad():
            audio_tensor = torch.FloatTensor(audio_np).unsqueeze(0).to(device)
            mel = get_mel_spectrogram(audio_tensor, model.h)
            generated = model(mel)
            generated_np = generated.squeeze().cpu().numpy()
            
            # Apply clarity and warmth adjustments
            if clarity != 0.5:
                from scipy import signal
                if clarity > 0.5:
                    sos = signal.butter(2, 8000, 'highpass', fs=24000, output='sos')
                    highs = signal.sosfilt(sos, generated_np)
                    blend = (clarity - 0.5) * 0.2
                    generated_np = generated_np * (1 - blend) + highs * blend
            
            if warmth != 0.7:
                from scipy import signal
                if warmth > 0.7:
                    sos = signal.butter(2, 400, 'lowpass', fs=24000, output='sos')
                    lows = signal.sosfilt(sos, generated_np)
                    blend = (warmth - 0.7) * 0.15
                    generated_np = generated_np * (1 - blend) + lows * blend
            
            # Final normalization
            max_val = np.max(np.abs(generated_np))
            if max_val > 0.95:
                generated_np = generated_np * (0.95 / max_val)
        
        # Save processed audio
        file_id = str(uuid.uuid4())
        output_path = os.path.join(TEMP_DIR, f"{file_id}.wav")
        sf.write(output_path, generated_np, 24000)
        
        # Return URLs
        return JSONResponse({
            "status": "success",
            "download_url": f"/download/{file_id}",
            "preview_url": f"/audio/{file_id}",
            "model": "v11a3b_clearvoice_epoch_346",
            "clearvoice_applied": use_clearvoice and (clearvoice_model is not None),
            "filename": f"roadshow_clearvoice_{file_id[:8]}.wav"
        })
        
    except Exception as e:
        print(f"Processing error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """Serve file for download"""
    file_path = os.path.join(TEMP_DIR, f"{file_id}.wav")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=f"roadshow_clearvoice_{file_id[:8]}.wav",
        headers={
            "Content-Disposition": f"attachment; filename=\"roadshow_clearvoice_{file_id[:8]}.wav\""
        }
    )

@app.get("/audio/{file_id}")
async def preview_file(file_id: str):
    """Serve audio file for playback"""
    file_path = os.path.join(TEMP_DIR, f"{file_id}.wav")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, media_type="audio/wav")

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "clearvoice_loaded": clearvoice_model is not None,
        "device": str(device) if device else "not initialized",
        "cuda_available": torch.cuda.is_available(),
        "checkpoint": "epoch_346",
        "val_loss": "9.02",
        "preprocessing": "ClearerVoice MossFormer2",
        "serving_mode": "file_urls"
    }

if __name__ == "__main__":
    print("="*60)
    print("Roadshow Studio v11a3b + CLEARERVOICE")
    print("Model: v11a3b_extended (epoch 346) with ClearerVoice denoising")
    print("="*60)
    
    # Run on port 8003 (different from main server on 8002)
    uvicorn.run(app, host="0.0.0.0", port=8003)