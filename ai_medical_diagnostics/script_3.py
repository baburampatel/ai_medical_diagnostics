# Create LLM provider files

# Base provider abstract class
base_provider = '''"""Abstract base class for LLM providers."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class LLMRequest(BaseModel):
    """Standard LLM request format."""
    messages: List[Dict[str, str]]
    temperature: float = 0.2
    max_tokens: int = 2000
    model: str = "gpt-3.5-turbo"

class LLMResponse(BaseModel):
    """Standard LLM response format."""
    content: str
    usage: Optional[Dict[str, Any]] = None
    model: str
    finish_reason: Optional[str] = None

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.endpoint = config.get("endpoint")
        self.model = config.get("model", "gpt-3.5-turbo")
        self.temperature = config.get("temperature", 0.2)
        self.max_tokens = config.get("max_tokens", 2000)
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.backoff_factor = config.get("backoff_factor", 2)
    
    @abstractmethod
    async def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> LLMResponse:
        """Generate completion from messages."""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate provider configuration."""
        pass
    
    def get_default_request(self, messages: List[Dict[str, str]]) -> LLMRequest:
        """Get default request object."""
        return LLMRequest(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model=self.model
        )
'''

# ChatGPTAPIFree provider implementation
chatgpt_api_free = '''"""ChatGPTAPIFree provider implementation."""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import BaseLLMProvider, LLMResponse
from ..utils.retries import async_retry_with_backoff
from ..utils.json_schemas import repair_json, validate_json_output

logger = logging.getLogger(__name__)

class ChatGPTAPIFreeProvider(BaseLLMProvider):
    """Provider for ChatGPTAPIFree service."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = httpx.AsyncClient(timeout=self.timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.aclose()
    
    def validate_config(self) -> bool:
        """Validate provider configuration."""
        required_fields = ["endpoint"]
        return all(field in self.config for field in required_fields)
    
    @async_retry_with_backoff(max_retries=3)
    async def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> LLMResponse:
        """Generate completion using ChatGPTAPIFree."""
        if not self.session:
            self.session = httpx.AsyncClient(timeout=self.timeout)
        
        # Prepare request payload
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }
        
        try:
            logger.info(f"Making request to {self.endpoint}")
            response = await self.session.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract content from response
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                
                # Attempt JSON repair if needed
                if kwargs.get("json_mode", False):
                    content = repair_json(content)
                
                return LLMResponse(
                    content=content,
                    usage=data.get("usage"),
                    model=data.get("model", self.model),
                    finish_reason=data["choices"][0].get("finish_reason")
                )
            else:
                raise ValueError("Invalid response format from API")
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            if e.response.status_code == 429:
                # Rate limit - wait longer before retry
                await asyncio.sleep(5)
            raise
        except httpx.TimeoutException:
            logger.error("Request timeout")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    async def generate_json_completion(
        self,
        messages: List[Dict[str, str]],
        schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate JSON completion with validation."""
        # Add JSON format instruction to system message
        json_instruction = "\\n\\nIMPORTANT: Respond only with valid JSON. Do not include any text before or after the JSON object."
        
        # Update system message
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += json_instruction
        else:
            messages.insert(0, {
                "role": "system", 
                "content": f"You are a medical AI assistant.{json_instruction}"
            })
        
        response = await self.generate_completion(messages, json_mode=True, **kwargs)
        
        try:
            json_data = json.loads(response.content)
            
            # Validate against schema if provided
            if schema:
                validate_json_output(json_data, schema)
            
            return json_data
        except json.JSONDecodeError:
            # Attempt repair and retry once
            repaired_content = repair_json(response.content)
            try:
                return json.loads(repaired_content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON: {response.content}")
                raise ValueError("Invalid JSON response from LLM")
'''

# OpenAI provider (optional)
openai_provider = '''"""OpenAI provider implementation."""
import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional
import httpx

from .base import BaseLLMProvider, LLMResponse
from ..utils.retries import async_retry_with_backoff

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI API."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = httpx.AsyncClient(timeout=self.timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.aclose()
    
    def validate_config(self) -> bool:
        """Validate provider configuration."""
        return self.api_key is not None
    
    @async_retry_with_backoff(max_retries=3)
    async def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> LLMResponse:
        """Generate completion using OpenAI API."""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
            
        if not self.session:
            self.session = httpx.AsyncClient(timeout=self.timeout)
        
        # Prepare request payload
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = await self.session.post(
                self.endpoint,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                
                return LLMResponse(
                    content=content,
                    usage=data.get("usage"),
                    model=data.get("model", self.model),
                    finish_reason=data["choices"][0].get("finish_reason")
                )
            else:
                raise ValueError("Invalid response format from OpenAI API")
                
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAI API error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI: {str(e)}")
            raise
'''

# Provider factory
provider_init = '''"""LLM provider factory and initialization."""
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .chatgpt_api_free import ChatGPTAPIFreeProvider
from .openai import OpenAIProvider
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load provider configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {
            "default_provider": "chatgpt_api_free",
            "providers": {
                "chatgpt_api_free": {
                    "endpoint": "https://chatgpt-api.shn.hk/v1/chat/completions",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.2,
                    "max_tokens": 2000,
                    "timeout": 30
                }
            }
        }

def get_provider(provider_name: Optional[str] = None, config_path: Optional[str] = None) -> BaseLLMProvider:
    """Get LLM provider instance."""
    config = load_config(config_path)
    
    if provider_name is None:
        provider_name = config.get("default_provider", "chatgpt_api_free")
    
    provider_config = config["providers"].get(provider_name)
    if not provider_config:
        raise ValueError(f"Provider '{provider_name}' not found in config")
    
    # Provider factory
    providers = {
        "chatgpt_api_free": ChatGPTAPIFreeProvider,
        "openai": OpenAIProvider
    }
    
    provider_class = providers.get(provider_name)
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    provider = provider_class(provider_config)
    
    if not provider.validate_config():
        raise ValueError(f"Invalid configuration for provider: {provider_name}")
    
    return provider
'''

# Write LLM provider files
with open("ai_medical_diagnostics/llm/providers/base.py", "w") as f:
    f.write(base_provider)

with open("ai_medical_diagnostics/llm/providers/chatgpt_api_free.py", "w") as f:
    f.write(chatgpt_api_free)

with open("ai_medical_diagnostics/llm/providers/openai.py", "w") as f:
    f.write(openai_provider)

with open("ai_medical_diagnostics/llm/providers/__init__.py", "w") as f:
    f.write(provider_init)

print("LLM provider files created successfully!")