"""S2: a one-layer toy world for intervention debugging.

The toy has artificial entities, four low-rank relation operators, and a
one-layer transformer that answers by retrieval.  Because the toy has one
decoder block, the intervention acts on the subject embedding or the
pre-attention input, so the keys and values computed inside the block change.
A post-block hook would not change later continuation and is not used here.

Relation ``r`` applies a rank-``k`` operator ``R_r = A_r B_r^T`` to the subject
embedding.  Low rank makes the intervention selective: each relation uses its
own subspace, and held-out entities generalize because the operator is shared.

The toy separates three things the real experiment needs:

1. factual competence on held-out entities before any intervention;
2. relation maps that damage the queried relation more than controls;
3. an external reliability probe and a reporting module, trained on one
   corruption family and tested on another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .interventions import attenuate_state, fit_relation_map, matched_random_delta


@dataclass
class ToyConfig:
    entities: int = 512
    relations: int = 4
    dim: int = 64
    heads: int = 4
    hidden: int = 128
    rank: int = 4
    train_entities: int = 384
    heldout_entities: int = 128
    steps: int = 800
    batch: int = 64
    lr: float = 1e-3
    seeds: int = 3
    alpha: float = 0.5


class ToyTransformer(nn.Module):
    """One decoder block with a linear readout to the entity space."""

    def __init__(self, config: ToyConfig):
        super().__init__()
        self.config = config
        self.entity_emb = nn.Embedding(config.entities, config.dim)
        self.relation_emb = nn.Embedding(config.relations, config.dim)
        self.position_emb = nn.Embedding(2, config.dim)
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = nn.MultiheadAttention(config.dim, config.heads, batch_first=True)
        self.ln2 = nn.LayerNorm(config.dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.dim),
        )

    def forward(
        self,
        entity_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        entity_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        subject = self.entity_emb(entity_ids) if entity_override is None else entity_override
        relation = self.relation_emb(relation_ids)
        positions = self.position_emb(torch.arange(2, device=subject.device))
        x = torch.stack([subject, relation], dim=1) + positions.unsqueeze(0)
        attended, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + attended
        x = x + self.mlp(self.ln2(x))
        return x[:, 1, :]  # answer vector at the relation position


class ToyWorld:
    """Data, model, relation maps, and corruption families for one seed."""

    def __init__(self, config: ToyConfig | None = None, seed: int = 0):
        self.config = config or ToyConfig()
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator = torch.Generator().manual_seed(seed)
        self.relation_matrices = []
        for _ in range(self.config.relations):
            a = torch.randn(self.config.dim, self.config.rank, generator=generator)
            b = torch.randn(self.config.dim, self.config.rank, generator=generator)
            self.relation_matrices.append(a @ b.T)
        self.model = ToyTransformer(self.config)
        self.train_ids = torch.arange(self.config.train_entities)
        self.heldout_ids = torch.arange(
            self.config.train_entities, self.config.train_entities + self.config.heldout_entities
        )
        self.relation_maps = {}

    # -- targets and training ----------------------------------------------
    def targets(self, entity_ids: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.model.entity_emb.weight.detach()
        subject = embeddings[entity_ids]
        matrix = torch.stack([self.relation_matrices[int(r)] for r in relation_ids])
        return torch.einsum("bd,bde->be", subject, matrix)

    def _batch(self):
        entity = self.train_ids[torch.randint(len(self.train_ids), (self.config.batch,))]
        relation = torch.randint(self.config.relations, (self.config.batch,))
        return entity, relation

    def train(self) -> float:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        for _ in range(self.config.steps):
            entity, relation = self._batch()
            target = self.targets(entity, relation)
            output = self.model(entity, relation)
            loss = F.mse_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return self.accuracy()

    @torch.no_grad()
    def success_mask(
        self,
        entity_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        output: torch.Tensor | None = None,
        entity_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Retrieval success: the output matches the relation target best."""

        target = self.targets(entity_ids, relation_ids)
        if output is None:
            output = self.model(entity_ids, relation_ids, entity_override=entity_override)
        embeddings = self.model.entity_emb.weight
        target_similarity = F.cosine_similarity(output, target, dim=-1)
        entity_similarity = F.cosine_similarity(
            output.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
        ).max(dim=-1).values
        return target_similarity > entity_similarity

    @torch.no_grad()
    def accuracy(self, entity_ids: torch.Tensor | None = None) -> float:
        entity_ids = self.heldout_ids if entity_ids is None else entity_ids
        correct = 0
        total = 0
        for relation in range(self.config.relations):
            relation_ids = torch.full_like(entity_ids, relation)
            success = self.success_mask(entity_ids, relation_ids)
            correct += int(success.sum())
            total += len(entity_ids)
        return correct / max(1, total)

    # -- relation maps ------------------------------------------------------
    def fit_maps(self, ridge: float = 1e-3) -> None:
        embeddings = self.model.entity_emb.weight.detach()
        states = embeddings[self.train_ids]
        for relation in range(self.config.relations):
            relation_ids = torch.full((len(self.train_ids),), relation)
            answers = self.targets(self.train_ids, relation_ids)
            self.relation_maps[relation] = fit_relation_map(
                states, answers, relation=str(relation), rank=self.config.rank, ridge=ridge
            )

    def corrupt(
        self,
        entity_ids: torch.Tensor,
        queried_relation: int,
        state: str,
        alpha: float | None = None,
        seed: int = 0,
    ) -> torch.Tensor:
        alpha = self.config.alpha if alpha is None else alpha
        clean = self.model.entity_emb.weight.detach()[entity_ids]
        if state == "clean":
            return clean
        relevant = self.relation_maps[queried_relation]
        if state == "relevant":
            return attenuate_state(clean, relevant, alpha)
        other = self.relation_maps[(queried_relation + 1) % self.config.relations]
        if state == "other":
            return attenuate_state(clean, other, alpha)
        if state == "random":
            delta, _ = matched_random_delta(clean, relevant, alpha, seed)
            return clean - delta
        raise ValueError(f"unknown state: {state}")

    # -- probe and report ---------------------------------------------------
    def collect_family(
        self,
        family: str,
        entity_ids: torch.Tensor,
        alpha: float | None = None,
    ) -> dict[str, np.ndarray]:
        states = []
        success = []
        subject_states = []
        for relation in range(self.config.relations):
            relation_ids = torch.full_like(entity_ids, relation)
            corrupted = self.corrupt(entity_ids, relation, family, alpha)
            success.append(
                self.success_mask(entity_ids, relation_ids, entity_override=corrupted)
                .float()
                .numpy()
            )
            subject_states.append(corrupted.detach().numpy())
            states.append(np.full(len(entity_ids), family))
        return {
            "subject_state": np.concatenate(subject_states, axis=0),
            "success": np.concatenate(success, axis=0),
            "family": np.concatenate(states, axis=0),
        }

    @staticmethod
    def probe_auroc(features: np.ndarray, success: np.ndarray) -> float | None:
        if len(np.unique(success)) < 2:
            return None
        model = LogisticRegression(max_iter=2000, C=0.1)
        model.fit(features, success.astype(int))
        return float(roc_auc_score(success.astype(int), model.predict_proba(features)[:, 1]))


def run_toy(config: ToyConfig | None = None) -> dict:
    """Train the toy and report competence, damage, and probe transfer."""

    config = config or ToyConfig()
    results = {}
    for seed in range(config.seeds):
        world = ToyWorld(config, seed=seed)
        accuracy = world.train()
        world.fit_maps()
        clean = world.collect_family("clean", world.heldout_ids)
        relevant = world.collect_family("relevant", world.heldout_ids)
        random = world.collect_family("random", world.heldout_ids)
        results[f"seed_{seed}"] = {
            "accuracy": accuracy,
            "clean_success": float(clean["success"].mean()),
            "relevant_success": float(relevant["success"].mean()),
            "random_success": float(random["success"].mean()),
            "probe_clean": world.probe_auroc(clean["subject_state"], clean["success"]),
            "probe_relevant": world.probe_auroc(
                relevant["subject_state"], relevant["success"]
            ),
            "probe_random": world.probe_auroc(random["subject_state"], random["success"]),
        }
    return results
