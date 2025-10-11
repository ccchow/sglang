from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

from fastapi import HTTPException, Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.entrypoints.openai.protocol import (
    DEFAULT_MODEL_NAME,
    ErrorResponse,
    OpenAIServingRequest,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


# Base class for specific endpoint handlers
class OpenAIServingBase(ABC):
    """Abstract base class for OpenAI endpoint handlers"""

    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager
        self.allowed_custom_labels = (
            set(
                self.tokenizer_manager.server_args.tokenizer_metrics_allowed_custom_labels
            )
            if isinstance(self.tokenizer_manager.server_args, ServerArgs)
            and self.tokenizer_manager.server_args.tokenizer_metrics_allowed_custom_labels
            else None
        )

    async def handle_request(
        self, request: OpenAIServingRequest, raw_request: Request
    ) -> Union[Any, StreamingResponse, ErrorResponse]:
        """Handle the specific request type with common pattern"""
        try:
            await self._apply_lora_alias_if_needed(request)

            # Validate request
            error_msg = self._validate_request(request)
            if error_msg:
                return self.create_error_response(error_msg)

            # Convert to internal format
            adapted_request, processed_request = self._convert_to_internal_request(
                request, raw_request
            )

            # Note(Xinyuan): raw_request below is only used for detecting the connection of the client
            if hasattr(request, "stream") and request.stream:
                return await self._handle_streaming_request(
                    adapted_request, processed_request, raw_request
                )
            else:
                return await self._handle_non_streaming_request(
                    adapted_request, processed_request, raw_request
                )
        except HTTPException as e:
            return self.create_error_response(
                message=e.detail, err_type=str(e.status_code), status_code=e.status_code
            )
        except ValueError as e:
            return self.create_error_response(
                message=str(e),
                err_type="BadRequest",
                status_code=400,
            )
        except Exception as e:
            logger.exception(f"Error in request: {e}")
            return self.create_error_response(
                message=f"Internal server error: {str(e)}",
                err_type="InternalServerError",
                status_code=500,
            )

    async def _apply_lora_alias_if_needed(
        self, request: OpenAIServingRequest
    ) -> None:
        """Populate ``lora_path`` when clients specify a LoRA via the ``model`` field."""

        if not hasattr(request, "lora_path") or not hasattr(request, "model"):
            return

        # Respect explicit lora_path values.
        if getattr(request, "lora_path") is not None:
            return

        server_args = getattr(self.tokenizer_manager, "server_args", None)
        if not server_args or not getattr(server_args, "enable_lora", False):
            return

        requested_model = getattr(request, "model", None)
        if requested_model in (None, ""):
            return

        served_names = self._get_served_model_aliases()
        served_names.add(DEFAULT_MODEL_NAME)
        if requested_model in served_names:
            return

        registry = getattr(self.tokenizer_manager, "lora_registry", None)
        if registry is None or not hasattr(registry, "has_lora"):
            return

        try:
            has_lora = await registry.has_lora(requested_model)
        except Exception:  # pragma: no cover - defensive, should not happen.
            return

        if has_lora:
            logger.debug(
                "Resolved LoRA adapter '%s' from model field", requested_model
            )
            setattr(request, "lora_path", requested_model)

    def _get_served_model_aliases(self) -> set[str]:
        """Return names that should map to the base model."""

        aliases: set[str] = set()
        served_model = getattr(self.tokenizer_manager, "served_model_name", None)
        if served_model:
            aliases.update(
                name.strip() for name in served_model.split(",") if name.strip()
            )

        model_path = getattr(self.tokenizer_manager, "model_path", None)
        if model_path:
            aliases.add(model_path)

        return aliases

    @abstractmethod
    def _request_id_prefix(self) -> str:
        """Generate request ID based on request type"""
        pass

    def _generate_request_id_base(self, request: OpenAIServingRequest) -> Optional[str]:
        """Generate request ID based on request type"""
        return None

        # TODO(chang): the rid is used in io_strcut check and often violates `The rid should be a list` AssertionError
        # Temporarily return None in this function until the rid logic is clear.
        if rid := getattr(request, "rid", None):
            return rid

        return f"{self._request_id_prefix()}{uuid.uuid4().hex}"

    def _compute_extra_key(self, request: OpenAIServingRequest) -> Optional[str]:
        """Compute the final extra_key by concatenating cache_salt and extra_key if both are provided."""
        parts = []
        for key in ["cache_salt", "extra_key"]:
            value = getattr(request, key, None)
            if value:
                if not isinstance(value, str):
                    raise TypeError(
                        f"Value of {key} must be a string, but got {type(value).__name__}"
                    )
                parts.append(value)
        return "".join(parts) if parts else None

    @abstractmethod
    def _convert_to_internal_request(
        self,
        request: OpenAIServingRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, OpenAIServingRequest]:
        """Convert OpenAI request to internal format"""
        pass

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: OpenAIServingRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ErrorResponse, ORJSONResponse]:
        """Handle streaming request

        Override this method in child classes that support streaming requests.
        """
        return self.create_error_response(
            message=f"{self.__class__.__name__} does not support streaming requests",
            err_type="NotImplementedError",
            status_code=501,
        )

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: OpenAIServingRequest,
        raw_request: Request,
    ) -> Union[Any, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming request

        Override this method in child classes that support non-streaming requests.
        """
        return self.create_error_response(
            message=f"{self.__class__.__name__} does not support non-streaming requests",
            err_type="NotImplementedError",
            status_code=501,
        )

    def _validate_request(self, _: OpenAIServingRequest) -> Optional[str]:
        """Validate request"""
        pass

    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
        param: Optional[str] = None,
    ) -> ORJSONResponse:
        """Create an error response"""
        # TODO: remove fastapi dependency in openai and move response handling to the entrypoint
        error = ErrorResponse(
            object="error",
            message=message,
            type=err_type,
            param=param,
            code=status_code,
        )
        return ORJSONResponse(content=error.model_dump(), status_code=status_code)

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
    ) -> str:
        """Create a streaming error response"""
        error = ErrorResponse(
            object="error",
            message=message,
            type=err_type,
            param=None,
            code=status_code,
        )
        return json.dumps({"error": error.model_dump()})

    def extract_custom_labels(self, raw_request):
        if (
            not self.allowed_custom_labels
            or not self.tokenizer_manager.server_args.tokenizer_metrics_custom_labels_header
        ):
            return None

        custom_labels = None
        header = (
            self.tokenizer_manager.server_args.tokenizer_metrics_custom_labels_header
        )
        try:
            raw_labels = (
                json.loads(raw_request.headers.get(header))
                if raw_request and raw_request.headers.get(header)
                else None
            )
        except json.JSONDecodeError as e:
            logger.exception(f"Error in request: {e}")
            raw_labels = None

        if isinstance(raw_labels, dict):
            custom_labels = {
                label: value
                for label, value in raw_labels.items()
                if label in self.allowed_custom_labels
            }
        return custom_labels
