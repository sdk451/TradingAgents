# TradingAgents/graph/signal_processing.py

import re
from typing import Union, Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Type alias for supported LLM types
LLMType = Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI]

# Maximum payload size to avoid embedding errors (36KB limit, use 30KB to be safe)
MAX_PAYLOAD_SIZE = 30000


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: LLMType):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def _extract_decision_regex(self, text: str) -> Optional[str]:
        """
        Try to extract the decision using regex patterns.
        
        Args:
            text: Trading signal text
            
        Returns:
            Extracted decision (BUY, SELL, or HOLD) or None if not found
        """
        # Pattern 1: "FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**"
        pattern1 = r"FINAL\s+TRANSACTION\s+PROPOSAL:\s*\*\*(BUY|SELL|HOLD)\*\*"
        match = re.search(pattern1, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Pattern 2: "**BUY/HOLD/SELL**" (standalone)
        pattern2 = r"\*\*(BUY|SELL|HOLD)\*\*"
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Pattern 3: "Recommendation: BUY/HOLD/SELL" or similar
        pattern3 = r"(?:recommendation|decision|recommend|conclusion)[:\s]+(BUY|SELL|HOLD)"
        match = re.search(pattern3, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        return None

    def _truncate_signal(self, signal: str, max_size: int = MAX_PAYLOAD_SIZE) -> str:
        """
        Truncate signal to fit within payload size limits.
        Prefer keeping the end of the signal where decisions usually appear.
        
        Args:
            signal: Full trading signal text
            max_size: Maximum size in bytes
            
        Returns:
            Truncated signal text
        """
        # Convert to bytes to check size
        signal_bytes = signal.encode('utf-8')
        
        if len(signal_bytes) <= max_size:
            return signal
        
        # Keep the last portion where decisions typically appear
        # Use 80% of max_size for the tail, 20% for context from beginning
        tail_size = int(max_size * 0.8)
        head_size = max_size - tail_size
        
        # Get bytes from end
        tail_bytes = signal_bytes[-tail_size:]
        # Try to decode from a valid UTF-8 boundary
        try:
            tail = tail_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If decoding fails, skip some bytes and try again
            tail = tail_bytes[1:].decode('utf-8', errors='ignore')
        
        # Get a small context from the beginning
        if head_size > 0:
            head_bytes = signal_bytes[:head_size]
            try:
                head = head_bytes.decode('utf-8')
            except UnicodeDecodeError:
                head = head_bytes.decode('utf-8', errors='ignore')
            return f"{head}\n\n[... content truncated ...]\n\n{tail}"
        
        return tail

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted decision (BUY, SELL, or HOLD)
        """
        # First, try to extract using regex (fast and doesn't require LLM call)
        decision = self._extract_decision_regex(full_signal)
        if decision:
            return decision
        
        # If regex extraction fails, use LLM but truncate if needed
        truncated_signal = self._truncate_signal(full_signal)
        
        messages = [
            (
                "system",
                "You are an efficient assistant designed to analyze paragraphs or financial reports provided by a group of analysts. Your task is to extract the investment decision: SELL, BUY, or HOLD. Provide only the extracted decision (SELL, BUY, or HOLD) as your output, without adding any additional text or information.",
            ),
            ("human", truncated_signal),
        ]

        return str(self.quick_thinking_llm.invoke(messages).content)
