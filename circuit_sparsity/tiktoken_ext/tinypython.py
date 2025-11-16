from __future__ import annotations

import os

from tiktoken.load import data_gym_to_mergeable_bpe_ranks

from circuit_sparsity.registries import MODEL_BASE_DIR

# Minimal set of special tokens referenced by this encoding.
ENDOFTEXT = "<|endoftext|>"


def _paths_from_env_or_default() -> tuple[str, str]:
    """Resolve encoder.json and vocab.bpe locations.

    If TINYPYTHON_TOK_DIR is set, use that local directory.
    Otherwise fall back to the original blob storage locations.
    """

    local_dir = os.environ.get("TINYPYTHON_TOK_DIR") or os.path.expanduser(
        os.path.join(MODEL_BASE_DIR, "tinypython")
    )
    vocab_bpe = os.path.join(local_dir, "vocab.bpe")
    encoder_json = os.path.join(local_dir, "encoder.json")
    return vocab_bpe, encoder_json


def tinypython_2k():
    """Return a tiktoken encoding spec for the tinypython_2k BPE.

    This mirrors the definition used internally, but is kept minimal and
    self-contained. It requires `blobfile` if reading from blob storage URLs.
    To force local files, set TINYPYTHON_TOK_DIR to a directory containing
    encoder.json and vocab.bpe.
    """

    vocab_bpe_file, encoder_json_file = _paths_from_env_or_default()
    mergeable_ranks = data_gym_to_mergeable_bpe_ranks(
        vocab_bpe_file=vocab_bpe_file,
        encoder_json_file=encoder_json_file,
    )

    return {
        "name": "tinypython_2k",
        # Same regex as the original definition
        "pat_str": r"""[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {ENDOFTEXT: 2047},
    }


# tiktoken discovers plugins via modules under `tiktoken_ext.*` that define
# ENCODING_CONSTRUCTORS (and optionally MODEL_TO_ENCODING). The keys are names
# accepted by `tiktoken.get_encoding(name)`.
ENCODING_CONSTRUCTORS = {
    # Each callable should return a spec mapping with keys:
    #   name, pat_str, mergeable_ranks, special_tokens
    "tinypython_2k": tinypython_2k,
}
