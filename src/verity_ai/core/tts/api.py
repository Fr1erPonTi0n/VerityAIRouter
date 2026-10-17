from typing import Any

import requests

from verity_ai.core.tts.voice import VoiceProvider


class TTSAPIError(RuntimeError):
    pass


class APIVoiceProvider(VoiceProvider):
    url: str = ""
    api_key: str = ""
    timeout: float = 30.0
    trust_env: bool = False

    @property
    def session(self) -> requests.Session:
        if not hasattr(self, "_session"):
            self._session = requests.Session()
            self._session.trust_env = self.trust_env

        return self._session

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "audio/wav",
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def _url(self, path: str) -> str:
        return f"{self.url.rstrip('/')}/{path.lstrip('/')}"

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        headers = {**self._headers(), **kwargs.pop("headers", {})}

        try:
            response = self.session.request(
                method,
                url,
                headers=headers,
                timeout=self.timeout,
                **kwargs,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            if exc.response is not None:
                detail = exc.response.text
            else:
                detail = str(exc)

            raise TTSAPIError(f"Ошибка TTS API: {detail}") from exc

        return response

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
    ) -> requests.Response:
        return self._request("POST", url, json=payload)

    def _get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        return self._request("GET", url, params=params).json()
