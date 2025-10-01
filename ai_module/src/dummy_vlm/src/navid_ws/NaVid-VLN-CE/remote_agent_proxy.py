import requests
import numpy as np
from typing import Any, Dict, Optional

def _to_list_if_np(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _to_list_if_np(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_list_if_np(v) for v in x]
    return x

class RemoteNaVidAgentProxy:
    def __init__(self, host: str, port: int = 3397, timeout: int = 60):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"Content-Type": "application/json"})

    def reset(self) -> None:
        r = self.s.post(f"{self.base}/reset", json={}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def act(self, observations, info, episode_id) -> Dict[str, Any]:
        obs_payload = {"rgb": _to_list_if_np(observations["rgb"]),
                       "instruction": {"text": str(observations["instruction"]["text"])}}
        info_payload = _to_list_if_np(info) if info is not None else None

        body = {"observations": obs_payload,
                "info": info_payload,
                "episode_id": episode_id}
        
        r = self.s.post(f"{self.base}/act", json=body, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["result"]
