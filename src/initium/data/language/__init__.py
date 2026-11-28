"""Broad language and learned world-state training utilities.

This package is deliberately separate from ``workspace_capability``: that
package remains a controlled episodic benchmark, while this package holds the
corpus, tokenizer, state-linked examples, objectives, dialogue, and evaluation
interfaces for the standalone language run.
"""

from initium.data.language.config import LanguageRunConfig

__all__ = ["LanguageRunConfig"]
