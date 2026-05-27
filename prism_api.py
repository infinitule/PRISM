"""
prism_api.py — T4: FastAPI REST Inference Endpoint
====================================================
Expose PRISM coherent fiber NN inference over HTTP.
Zero new dependencies beyond numpy + fastapi + uvicorn.

Endpoints:
  POST /infer               — run inference on a feature vector
  GET  /health              — server health + model registry
  GET  /models              — list all loaded use-case models
  POST /train/{use_case}    — train a model for a specific use case
  POST /domain_shift        — detect domain shift in a new batch

Run:
  uvicorn prism_api:app --reload --port 8000

Test:
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/infer \
       -H "Content-Type: application/json" \
       -d '{"model":"mic","features":[2.0,14.0,0.85,-7.0,0.1,3.0,10.0,1.0]}'

Simulation mode (no uvicorn):
  from prism_api import PRISMApp
  app = PRISMApp()
  result = app.infer("mic", [2.0, 14.0, 0.85, -7.0, 0.1, 3.0, 10.0, 1.0])
"""

import numpy as np
import time
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from recursive_prompt import MetaMetaPrompt
from use_cases import (
    generate_mic_dataset, generate_amc_dataset,
    generate_fde_dataset, generate_dci_dataset,
    MIC_CLASSES, MIC_FEATURES,
    AMC_CLASSES, AMC_FEATURES,
    FDE_CLASSES, FDE_FEATURES,
    DCI_CLASSES, DCI_FEATURES,
    _train_and_transfer, detect_domain_shift,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Registry of available models (populated lazily on first /train or /infer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_USE_CASE_CONFIGS = {
    'mic': {
        'name':       'Multi-Impairment Classifier',
        'classes':    MIC_CLASSES,
        'features':   MIC_FEATURES,
        'arch':       [8, 20, 10, 5],
        'gen_fn':     lambda: generate_mic_dataset(n=300),
        'epochs':     600,
    },
    'amc': {
        'name':       'Adaptive Modulation Controller',
        'classes':    AMC_CLASSES,
        'features':   AMC_FEATURES,
        'arch':       [8, 24, 12, 6],
        'gen_fn':     lambda: generate_amc_dataset(n=360),
        'epochs':     600,
    },
    'fde': {
        'name':       'OTDR Fault Diagnosis Engine',
        'classes':    FDE_CLASSES,
        'features':   FDE_FEATURES,
        'arch':       [8, 20, 10, 5],
        'gen_fn':     lambda: generate_fde_dataset(n=300),
        'epochs':     800,
    },
    'dci': {
        'name':       'DCI Link State Classifier',
        'classes':    DCI_CLASSES,
        'features':   DCI_FEATURES,
        'arch':       [8, 20, 10, 5],
        'gen_fn':     lambda: generate_dci_dataset(n=300),
        'epochs':     700,
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PRISMApp:
    """
    Pure-Python PRISM application object.
    Works standalone (no FastAPI) for testing and simulation mode.
    The FastAPI app delegates to this class.
    """

    def __init__(self, rng_seed: int = 42):
        self.mp      = MetaMetaPrompt(seed_dim=64, max_depth=16, rng_seed=rng_seed)
        self._models: dict[str, dict] = {}   # {use_case: {nn, acc, th_acc, classes, features}}
        self._start_time = time.time()

    # ── Model management ───────────────────────────────────────────────────
    def train(self, use_case: str, verbose: bool = False) -> dict:
        """Train a model for the given use case. Returns training summary."""
        uc = use_case.lower()
        if uc not in _USE_CASE_CONFIGS:
            raise ValueError(f"Unknown use case '{uc}'. "
                             f"Valid: {list(_USE_CASE_CONFIGS)}")
        cfg = _USE_CASE_CONFIGS[uc]
        X, Y = cfg['gen_fn']()
        split = int(0.8 * len(X))
        X_tr, Y_tr = X[:split], Y[:split]
        X_te, Y_te = X[split:], Y[split:]

        t0 = time.time()
        nn, acc, th_acc, trainer = _train_and_transfer(
            cfg['arch'], X_tr, Y_tr, X_te, Y_te,
            epochs=cfg['epochs'], mp=self.mp, verbose=verbose)
        elapsed = time.time() - t0

        self._models[uc] = {
            'nn':       nn,
            'trainer':  trainer,
            'X_tr':     X_tr,
            'classes':  cfg['classes'],
            'features': cfg['features'],
            'acc':      acc,
            'th_acc':   th_acc,
            'arch':     cfg['arch'],
            'name':     cfg['name'],
            'trained_at': time.time(),
        }
        return {
            'use_case':    uc,
            'name':        cfg['name'],
            'fiber_acc':   round(acc, 4),
            'theory_acc':  round(th_acc, 4),
            'arch':        cfg['arch'],
            'train_time_s': round(elapsed, 2),
            'wdm_channels': nn.total_lambda_channels(),
        }

    def _ensure_model(self, use_case: str) -> dict:
        uc = use_case.lower()
        if uc not in self._models:
            self.train(uc, verbose=False)
        return self._models[uc]

    # ── Inference ──────────────────────────────────────────────────────────
    def infer(self, use_case: str,
              features: list[float] | np.ndarray) -> dict:
        """
        Run inference. Auto-trains model on first call.

        Returns:
            predicted_class : string class name
            predicted_idx   : integer class index
            confidence      : probability of predicted class
            probabilities   : {class_name: probability} dict
            latency_us      : inference latency in microseconds
            model           : use case identifier
        """
        model = self._ensure_model(use_case)
        x     = np.asarray(features, dtype=float)
        t0    = time.perf_counter()
        probs = model['nn'].forward(x)
        lat   = (time.perf_counter() - t0) * 1e6

        pred_idx   = int(np.argmax(probs))
        classes    = model['classes']
        return {
            'predicted_class': classes[pred_idx],
            'predicted_idx':   pred_idx,
            'confidence':      float(probs[pred_idx]),
            'probabilities':   {c: float(p) for c, p in zip(classes, probs)},
            'latency_us':      round(lat, 2),
            'model':           use_case.lower(),
            'substrate':       'coherent IQ · 1550 nm · DWDM',
        }

    # ── Domain shift ───────────────────────────────────────────────────────
    def check_domain_shift(self, use_case: str,
                           X_new: list[list[float]]) -> dict:
        """Run domain-shift detection against training distribution."""
        model = self._ensure_model(use_case)
        X_new_arr = np.asarray(X_new, dtype=float)
        result    = detect_domain_shift(model['X_tr'], X_new_arr)
        result['use_case'] = use_case.lower()
        return result

    # ── Health / registry ──────────────────────────────────────────────────
    def health(self) -> dict:
        return {
            'status':       'ok',
            'uptime_s':     round(time.time() - self._start_time, 1),
            'loaded_models': list(self._models.keys()),
            'available_models': list(_USE_CASE_CONFIGS.keys()),
            'substrate':    'PRISM coherent IQ fiber NN · 1550 nm',
        }

    def models(self) -> list[dict]:
        result = []
        for uc, cfg in _USE_CASE_CONFIGS.items():
            loaded = uc in self._models
            entry = {
                'id':       uc,
                'name':     cfg['name'],
                'loaded':   loaded,
                'classes':  cfg['classes'],
                'n_features': len(cfg['features']),
                'arch':     cfg['arch'],
            }
            if loaded:
                m = self._models[uc]
                entry['fiber_acc']   = round(m['acc'], 4)
                entry['theory_acc']  = round(m['th_acc'], 4)
                entry['wdm_channels'] = m['nn'].total_lambda_channels()
            result.append(entry)
        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI app (only constructed when fastapi is importable)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    _prism = PRISMApp()
    app    = FastAPI(
        title="PRISM Coherent Fiber NN API",
        description="Photonic Recursive Intelligence with Synaptic Memory — REST inference layer",
        version="1.0.0",
    )

    # ── Request/response models ────────────────────────────────────────────
    class InferRequest(BaseModel):
        model:    str
        features: list[float]

    class DomainShiftRequest(BaseModel):
        use_case: str
        batch:    list[list[float]]

    # ── Endpoints ──────────────────────────────────────────────────────────
    @app.get("/health")
    def health():
        return _prism.health()

    @app.get("/models")
    def models():
        return _prism.models()

    @app.post("/infer")
    def infer(req: InferRequest):
        try:
            return _prism.infer(req.model, req.features)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/train/{use_case}")
    def train(use_case: str):
        try:
            return _prism.train(use_case, verbose=False)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/domain_shift")
    def domain_shift(req: DomainShiftRequest):
        try:
            return _prism.check_domain_shift(req.use_case, req.batch)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/")
    def root():
        return JSONResponse({
            "name":    "PRISM API",
            "version": "1.0.0",
            "docs":    "/docs",
            "health":  "/health",
            "models":  "/models",
        })

except ImportError:
    # FastAPI not installed — PRISMApp still works for simulation
    app = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    # Simulation mode — no HTTP server needed
    print("\n  PRISM API — simulation mode (no uvicorn required)")
    prism = PRISMApp()

    print("\n  Training MIC model...")
    summary = prism.train('mic', verbose=False)
    print(f"  Train result: {summary}")

    print("\n  Running inference...")
    result = prism.infer('mic', [2.0, 14.0, 0.85, -7.0, 0.1, 3.0, 10.0, 1.0])
    print(f"  Inference: {result['predicted_class']}  "
          f"(conf={result['confidence']:.3f}  lat={result['latency_us']:.1f} µs)")

    print("\n  Health check:")
    h = prism.health()
    for k, v in h.items():
        print(f"    {k}: {v}")

    print("\n  Registered models:")
    for m in prism.models():
        status = f"acc={m['fiber_acc']:.1%}" if m['loaded'] else "not loaded"
        print(f"    {m['id']:>6}  {m['name']:<40}  {status}")
