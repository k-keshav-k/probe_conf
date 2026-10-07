"""Model loading, hook contracts, and collection."""

from .collect import (  # noqa: F401
    CollectSettings,
    ModelWeights,
    collect,
    collect_track_a,
    collect_track_b,
    extract_features_for_sequences,
    parse_confidence,
)
from .contract import MLPCapture, get_layers, mlp_modules, verify_mlp_contract  # noqa: F401
from .loader import (  # noqa: F401
    LoadedModel,
    chat_prompt_ids,
    chat_prompt_text,
    confidence_prompt_ids,
    load_model_and_tokenizer,
)
