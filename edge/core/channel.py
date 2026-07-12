# core/channel.py
import time
import msgpack
import requests
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod


class BaseChannel(ABC):
    @abstractmethod
    def submit(self, endpoint_url: str, data: bytes, headers: dict = None, tag: str = None):
        pass

    @abstractmethod
    def drain_tag(self, tag: str):
        pass

    @abstractmethod
    def close(self):
        pass


class NetworkChannel(BaseChannel):
    """Production HTTP channel from the Edge host to the Cloud verifier."""

    def __init__(self, config):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.session = requests.Session()
        if not getattr(self.config, "use_env_proxy", True):
            self.session.trust_env = False

    def _simulated_send(self, url: str, data: bytes, headers: dict) -> dict:
        size_mb = len(data) / (1024 * 1024)
        upload_delay = size_mb / self.config.bandwidth_MBps if self.config.bandwidth_MBps > 0 else 0
        one_way_latency = self.config.base_latency_c / 2 if self.config.base_latency_c > 0 else 0
        time.sleep(upload_delay + one_way_latency)

        try:
            timeout = getattr(self.config, "timeout_s", 120)
            response = self.session.post(url, data=data, headers=headers, timeout=timeout)
            if response.headers.get("Content-Type", "").split(";", 1)[0] == "application/msgpack":
                result = msgpack.unpackb(response.content, raw=False)
            else:
                result = response.json()
            if response.status_code >= 400 and not isinstance(result, dict):
                result = {"error": f"Cloud returned HTTP {response.status_code}: {result}"}
            elif response.status_code >= 400 and "error" not in result:
                result["error"] = f"Cloud returned HTTP {response.status_code}"
        except Exception as exc:
            result = {"error": str(exc)}

        if one_way_latency > 0:
            time.sleep(one_way_latency)
        return result

    def submit(self, endpoint_url, data, headers=None, tag=None):
        headers = headers or {"Content-Type": "application/octet-stream"}
        return self.executor.submit(self._simulated_send, endpoint_url, data, headers)

    def drain_tag(self, tag):
        return []

    def close(self):
        self.executor.shutdown(wait=True)
        self.session.close()
