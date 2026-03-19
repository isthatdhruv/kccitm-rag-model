"""Token budget manager for the LLM context window.

Manages the 32K context window budget (Qwen 3 8B). Ensures we never exceed
the LLM's context limit by tracking token usage across all components.

Budget allocation:
- System prompt:      ~800 tokens
- Chat history:       ~3000 tokens (sliding window)
- History summary:    ~500 tokens (condensed older messages — Phase 6)
- RAG context:        ~4000 tokens (retrieved chunks)
- SQL results:        ~1000 tokens
- Generation buffer:  ~2500 tokens (space for LLM output)
"""
from __future__ import annotations

import tiktoken


class ContextBuilder:
    """Token-aware context assembly for LLM prompts."""

    MAX_CONTEXT_TOKENS = 32000
    SYSTEM_PROMPT_BUDGET = 800
    CHAT_HISTORY_BUDGET = 3000
    HISTORY_SUMMARY_BUDGET = 500
    RAG_CONTEXT_BUDGET = 4000
    SQL_RESULTS_BUDGET = 1000
    GENERATION_BUFFER = 2500

    def __init__(self):
        # cl100k_base approximates Qwen's tokenizer well enough for budgeting
        self.encoder = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Token utilities
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within a token budget."""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoder.decode(tokens[:max_tokens]) + "\n... (truncated)"

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    def build_rag_context(
        self,
        chunks: list[dict],
        max_tokens: int | None = None,
    ) -> str:
        """
        Build RAG context from retrieved chunks, fitting within budget.

        Adds chunks one by one until budget is exhausted.
        Each chunk gets a metadata header line + the chunk text.
        """
        max_tokens = max_tokens or self.RAG_CONTEXT_BUDGET
        context_parts: list[str] = []
        total_tokens = 0

        header = f"RETRIEVED STUDENT DATA ({len(chunks)} records):\n"
        total_tokens += self.count_tokens(header)
        context_parts.append(header)

        for i, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            entry_header = (
                f"[{i + 1}] Student: {meta.get('name', 'N/A')} | "
                f"Roll: {meta.get('roll_no', 'N/A')} | "
                f"Branch: {meta.get('branch', 'N/A')} | "
                f"Sem: {meta.get('semester', 'N/A')} | "
                f"SGPA: {meta.get('sgpa', 'N/A')}"
            )
            entry_text = chunk.get("text", "")
            entry = f"\n{entry_header}\n{entry_text}\n"

            entry_tokens = self.count_tokens(entry)

            if total_tokens + entry_tokens > max_tokens:
                remaining = len(chunks) - i
                context_parts.append(
                    f"\n... and {remaining} more relevant records (truncated for context length)"
                )
                break

            context_parts.append(entry)
            total_tokens += entry_tokens

        return "".join(context_parts)

    def build_sql_context(self, sql_result) -> str:
        """
        Build SQL result context, fitting within budget.

        Args:
            sql_result: SQLResult from the SQL pipeline
        """
        if not sql_result or not sql_result.success:
            return ""

        parts = ["SQL QUERY RESULTS:\n"]
        parts.append(f"Query: {sql_result.sql}\n")
        parts.append(f"Explanation: {sql_result.explanation}\n")

        if sql_result.formatted_table:
            table_text = sql_result.formatted_table
            if self.count_tokens(table_text) > self.SQL_RESULTS_BUDGET:
                table_text = self.truncate_to_budget(table_text, self.SQL_RESULTS_BUDGET)
            parts.append(f"\n{table_text}\n")
        elif sql_result.formatted_text:
            parts.append(f"\n{sql_result.formatted_text}\n")

        return "".join(parts)

    # ------------------------------------------------------------------
    # Chat history management
    # ------------------------------------------------------------------

    def trim_chat_history(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
    ) -> list[dict]:
        """
        Trim chat history to fit within budget using sliding window.

        Strategy:
        - Always keep the last 8 messages (4 user/assistant pairs)
        - If history exceeds budget, drop oldest messages
        - In Phase 6, older messages get summarized instead of dropped
        """
        max_tokens = max_tokens or self.CHAT_HISTORY_BUDGET
        if not messages:
            return []

        keep_last = 8
        if len(messages) <= keep_last:
            return messages

        recent = messages[-keep_last:]
        recent_tokens = sum(self.count_tokens(m.get("content", "")) for m in recent)

        if recent_tokens >= max_tokens:
            # Even recent messages exceed budget — return what fits
            trimmed = []
            token_count = 0
            for msg in reversed(recent):
                msg_tokens = self.count_tokens(msg.get("content", ""))
                if token_count + msg_tokens > max_tokens:
                    break
                trimmed.insert(0, msg)
                token_count += msg_tokens
            return trimmed

        # Add older messages from most recent to oldest within remaining budget
        remaining_budget = max_tokens - recent_tokens
        older = messages[:-keep_last]

        included_older = []
        for msg in reversed(older):
            msg_tokens = self.count_tokens(msg.get("content", ""))
            if remaining_budget - msg_tokens < 0:
                break
            included_older.insert(0, msg)
            remaining_budget -= msg_tokens

        return included_older + recent

    # ------------------------------------------------------------------
    # Budget estimation
    # ------------------------------------------------------------------

    def estimate_total_usage(
        self,
        system_prompt: str,
        chat_history: list[dict],
        rag_context: str,
        sql_context: str,
        query: str,
    ) -> dict:
        """
        Estimate total token usage across all context components.

        Returns dict with per-component token counts and total.
        """
        system_tokens = self.count_tokens(system_prompt)
        history_tokens = sum(
            self.count_tokens(m.get("content", "")) for m in (chat_history or [])
        )
        rag_tokens = self.count_tokens(rag_context)
        sql_tokens = self.count_tokens(sql_context)
        query_tokens = self.count_tokens(query)

        total = system_tokens + history_tokens + rag_tokens + sql_tokens + query_tokens

        return {
            "system_prompt": system_tokens,
            "chat_history": history_tokens,
            "rag_context": rag_tokens,
            "sql_context": sql_tokens,
            "query": query_tokens,
            "total_input": total,
            "remaining_for_generation": self.MAX_CONTEXT_TOKENS - total,
            "within_budget": total < (self.MAX_CONTEXT_TOKENS - self.GENERATION_BUFFER),
        }
