from __future__ import annotations

LOCAL_MODELS: dict[str, dict] = {
    "wan-2.2": {
        "type": "wan",
        "model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "height": 704,
        "width": 1280,
        "num_frames": 49,
        "fps": 24,
    },
    "cosmos-2.5": {
        "type": "cosmos2.5",
        "model_id": "nvidia/Cosmos-Predict2.5-2B",
        "revision": "diffusers/base/post-trained",
        "height": 704,
        "width": 1280,
        "num_frames": 33,
        "fps": 16,
    },
}


REPLICATE_MODELS: dict[str, dict] = {
    "veo-3.1": {
        "model_id": "google/veo-3.1",
        "image_param": "image",
        "duration_param": "duration",
        "duration": 4,
        "negative_param": "negative_prompt",
        "height": 720,
        "width": 1280,
        "fps": 24,
        "extra": {"resolution": "720p", "aspect_ratio": "16:9", "generate_audio": False},
    },
    "veo-3.1-fast": {
        "model_id": "google/veo-3.1-fast",
        "image_param": "image",
        "duration_param": "duration",
        "duration": 4,
        "negative_param": "negative_prompt",
        "height": 720,
        "width": 1280,
        "fps": 24,
        "extra": {"resolution": "720p", "aspect_ratio": "16:9", "generate_audio": False},
    },
    "kling-3.0": {
        "model_id": "kwaivgi/kling-v3-video",
        "image_param": "start_image",
        "duration_param": "duration",
        "duration": 3,
        "negative_param": "negative_prompt",
        "height": 720,
        "width": 1280,
        "fps": 24,
        "extra": {"mode": "pro", "aspect_ratio": "16:9", "generate_audio": False},
    },
    "seedance-2.0": {
        "model_id": "bytedance/seedance-2.0",
        "image_param": "image",
        "duration_param": "duration",
        "duration": 4,
        "negative_param": None,
        "height": 720,
        "width": 1280,
        "fps": 24,
        "extra": {"resolution": "720p", "aspect_ratio": "16:9", "generate_audio": False},
    },
    "seedance-2.0-fast": {
        "model_id": "bytedance/seedance-2.0-fast",
        "image_param": "image",
        "duration_param": "duration",
        "duration": 4,
        "negative_param": None,
        "height": 720,
        "width": 1280,
        "fps": 24,
        "extra": {"resolution": "720p", "aspect_ratio": "16:9", "generate_audio": False},
    },
    "grok-imagine": {
        "model_id": "xai/grok-imagine-video",
        "image_param": "image",
        "duration_param": "duration",
        "duration": 4,
        "negative_param": None,
        "height": 720,
        "width": 1280,
        "fps": 24,
        "extra": {"resolution": "720p", "aspect_ratio": "auto"},
    },
}


ALL_MODELS: dict[str, dict] = {**LOCAL_MODELS, **REPLICATE_MODELS}
