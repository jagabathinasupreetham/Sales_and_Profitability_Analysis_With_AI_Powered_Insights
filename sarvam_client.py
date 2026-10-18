"""Small Sarvam chat-completions client. API calls are optional for local testing."""

import requests

from config import SARVAM_API_KEY, SARVAM_BASE_URL, DEFAULT_MODEL


class SarvamError(RuntimeError):
    """Raised when a Sarvam API request fails or returns an invalid response."""

    pass


def chat_completion(messages, model=None, temperature=0.0, timeout=45):
    """Send a chat-completion request to Sarvam and return the answer."""

    # 1. Check whether the API key is configured.
    if not SARVAM_API_KEY or SARVAM_API_KEY == "your_sarvam_api_key_here":
        raise SarvamError(
            "Sarvam API key is missing. Add SARVAM_API_KEY to your .env file."
        )

    # 2. Prepare the request.
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    try:
        # 3. Send the request.
        response = requests.post(
            SARVAM_BASE_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )

        response.raise_for_status()

        # 4. Parse the response.
        data = response.json()

        # Print diagnostic information without exposing the API key.
        if not isinstance(data, dict):
            raise SarvamError(
                "Sarvam returned an unexpected response: "
                "the response is not a JSON object."
            )

        choices = data.get("choices")

        if not isinstance(choices, list) or not choices:
            # Avoid printing the complete response because it may contain
            # information that should not appear in the terminal.
            raise SarvamError(
                "Sarvam returned no choices. "
                f"Response fields received: {list(data.keys())}"
            )

        first_choice = choices[0]

        if not isinstance(first_choice, dict):
            raise SarvamError(
                "Sarvam returned an unexpected format for its first choice."
            )

        message = first_choice.get("message")

        if not isinstance(message, dict):
            raise SarvamError(
                "Sarvam's first choice does not contain a valid message. "
                f"Available fields: {list(first_choice.keys())}"
            )

        content = message.get("content")

        # 5. Handle empty or unexpected content.
        if isinstance(content, str):
            content = content.strip()

            if content:
                return content

        # Some APIs return content as a list of text blocks.
        if isinstance(content, list):
            text_parts = []

            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text")

                    if isinstance(text_value, str) and text_value.strip():
                        text_parts.append(text_value.strip())

            if text_parts:
                return "\n".join(text_parts)

        # Include safe response metadata to help diagnose the issue.
        finish_reason = first_choice.get("finish_reason")

        raise SarvamError(
            "Sarvam returned an empty or unsupported answer. "
            f"Content type: {type(content).__name__}; "
            f"finish reason: {finish_reason!r}; "
            f"response fields: {list(data.keys())}"
        )

    except requests.RequestException as exc:
        detail = ""

        if getattr(exc, "response", None) is not None:
            detail = (
                f" (HTTP {exc.response.status_code}: "
                f"{exc.response.text[:400]})"
            )

        raise SarvamError(
            f"Sarvam request failed{detail}"
        ) from exc

    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise SarvamError(
            f"Sarvam returned an unexpected response format: {exc}"
        ) from exc