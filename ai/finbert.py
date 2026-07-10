"""
ai/finbert.py — FinBERT Sentiment Analysis Module

PURPOSE:
    Expose a single function — analyze_sentiment(text) — that takes
    a piece of financial text and returns the predicted sentiment
    label and confidence score.

SINGLE RESPONSIBILITY:
    This module ONLY performs sentiment analysis.
    It does NOT scrape, clean text, or write files.
    It does NOT print anything — pure computation only.

LIBRARIES:
    transformers  → Hugging Face library for loading pre-trained NLP models.
                     AutoTokenizer converts text into token IDs that the
                     model understands. AutoModelForSequenceClassification
                     loads the FinBERT model with its classification head.

    torch         → PyTorch deep learning framework. FinBERT is a PyTorch
                     model. We use torch for tensor operations, softmax,
                     and inference (torch.no_grad disables gradient tracking
                     for faster, memory-efficient inference).

MODEL:
    ProsusAI/finbert — a BERT model fine-tuned on 10,000+ financial news
    articles. It classifies text into: positive, negative, or neutral.

WHY lazy loading?
    The model is ~420MB. Loading it on import would slow down every
    module that imports ai.finbert. Instead, we load on first call
    to analyze_sentiment() and cache it for subsequent calls.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

from config import FINBERT_MODEL_NAME, MAX_TOKEN_LENGTH


# ---------------------------------------------------------------------------
# Module-level cache for the model and tokenizer (lazy singleton pattern)
# ---------------------------------------------------------------------------
# These are None until the first call to analyze_sentiment().
# After that, they hold the loaded model and tokenizer so we
# don't re-download / re-load on every call.
_tokenizer = None
_model = None


def _load_model():
    """
    Load the FinBERT model and tokenizer from Hugging Face Hub.

    Uses the module-level cache (_tokenizer, _model) to ensure
    the model is loaded only once per process.

    WHY a separate function?
        → Keeps the loading logic isolated. If you later switch to
          a different model or add GPU support, you only change here.
    """
    global _tokenizer, _model

    # Skip loading if already cached
    if _tokenizer is not None and _model is not None:
        return

    # AutoTokenizer.from_pretrained() downloads the tokenizer config
    # and vocabulary files. It converts text → token IDs that the model
    # understands (e.g., "stock" → 4127).
    _tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)

    # AutoModelForSequenceClassification.from_pretrained() downloads the
    # model weights (~420MB on first run, cached afterwards in ~/.cache/huggingface).
    # It loads BERT + a classification head (3 output neurons: pos/neg/neutral).
    _model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)

    # model.eval() switches the model from training mode to inference mode.
    # This disables dropout layers, which would add randomness to outputs.
    _model.eval()


def analyze_sentiment(text: str) -> dict:
    """
    Analyze the financial sentiment of a given text using FinBERT.

    Args:
        text: A piece of financial text (headline, article excerpt, etc.).
              Should be pre-cleaned (no HTML tags or special characters).

    Returns:
        A dict with exactly two keys:
            - "sentiment"  (str):   One of "positive", "negative", "neutral"
            - "confidence" (float): Probability score (0.0 to 1.0) for the
                                     predicted sentiment class

    Example:
        >>> analyze_sentiment("Apple stock surged after strong earnings")
        {"sentiment": "positive", "confidence": 0.9503}

    WORKFLOW:
        1. Load model (lazy — only on first call)
        2. Tokenize the input text
        3. Run inference (forward pass through the model)
        4. Apply softmax to get probability distribution
        5. Pick the class with the highest probability
        6. Return sentiment label and confidence score
    """
    # ── Step 1: Ensure model is loaded ─────────────────────────
    _load_model()

    # ── Step 2: Tokenize the input text ────────────────────────
    # The tokenizer converts text into a dict of tensors:
    #   - input_ids:      token IDs (integers) for each word/subword
    #   - attention_mask:  1s for real tokens, 0s for padding
    # truncation=True cuts text longer than MAX_TOKEN_LENGTH tokens.
    # return_tensors="pt" returns PyTorch tensors (not lists).
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
        padding=True,
    )

    # ── Step 3: Run inference ──────────────────────────────────
    # torch.no_grad() disables gradient computation.
    # WHY? We're not training — just predicting. Disabling gradients
    # saves memory and speeds up the forward pass by ~20%.
    with torch.no_grad():
        outputs = _model(**inputs)

    # ── Step 4: Apply softmax to get probabilities ─────────────
    # outputs.logits is a tensor of raw scores (logits) for each class.
    # Shape: [1, 3] — one sample, three classes (pos/neg/neutral).
    # F.softmax converts logits to probabilities that sum to 1.0.
    # dim=-1 means apply softmax along the last dimension (classes).
    probabilities = F.softmax(outputs.logits, dim=-1)

    # ── Step 5: Get the predicted class ────────────────────────
    # torch.argmax returns the index of the highest probability.
    # .item() converts the single-element tensor to a Python int.
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    # FinBERT label mapping: index 0=positive, 1=negative, 2=neutral
    label_map = {0: "positive", 1: "negative", 2: "neutral"}
    sentiment = label_map[predicted_class]

    # ── Step 6: Get the confidence score ───────────────────────
    # probabilities[0] gets the first (and only) sample.
    # [predicted_class] gets the probability for the winning class.
    # .item() converts the tensor scalar to a Python float.
    # round(..., 4) keeps 4 decimal places for readability.
    confidence = round(probabilities[0][predicted_class].item(), 4)

    return {
        "sentiment": sentiment,
        "confidence": confidence,
    }
