from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import create_masked_tensor, get_last


class SASRecBackbone(nn.Module):
    """
    Self-Attentive Sequential Recommendation (SASRec) backbone architecture.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        num_heads: int = 2,
        max_seq_len: int = 512,
        dropout_rate: float = 0.2,
        num_transformer_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.item_embeddings = nn.Embedding(self.num_items, self.embedding_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, self.embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=num_heads,
            dropout=dropout_rate,
            batch_first=True,
            # norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_transformer_layers,
            enable_nested_tensor=False,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        lengths = inputs["lengths"]
        item_ids = inputs["item_id"]
        positions = inputs["positions"]

        embeddings = self.item_embeddings(item_ids)

        position_embeddings = self.position_embeddings(positions)

        embeddings = embeddings + position_embeddings

        embeddings, mask = create_masked_tensor(data=embeddings, lengths=lengths)

        causal_mask = torch.triu(
            torch.ones(
                (embeddings.shape[1], embeddings.shape[1]),
                device=embeddings.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        encoder_output = self.transformer_encoder(embeddings, mask=causal_mask, src_key_padding_mask=~mask)
        encoder_output = encoder_output[mask]

        return {"encoder_output": encoder_output}


class SASRecModel(nn.Module):
    """
    Complete SASRec recommendation model combining backbone encoder with training and inference logic.
    """

    def __init__(
        self,
        backbone: SASRecBackbone,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.init_weights(0.02)

    @torch.no_grad()
    def init_weights(self, initializer_range: float) -> None:
        for key, value in self.named_parameters():
            if "weight" in key:
                if "norm" in key:
                    nn.init.ones_(value.data)
                else:
                    nn.init.trunc_normal_(
                        value.data,
                        std=initializer_range,
                        a=-2 * initializer_range,
                        b=2 * initializer_range,
                    )
            else:
                if value.dim() > 0:
                    pass
                if "bias" in key:
                    nn.init.zeros_(value.data)

    def compute_loss(self, inputs: Dict, backbone_output: Dict[str, torch.Tensor]) -> Dict:
        raise NotImplementedError

    def forward(self, inputs: Dict):
        backbone_outputs = self.backbone(inputs["history"])

        if self.training:
            return {"loss": self.compute_loss(inputs, backbone_outputs)}
        else:
            last_embeddings = get_last(backbone_outputs["encoder_output"], inputs["history"]["lengths"])
            last_labels = get_last(inputs["labels"]["item_id"], inputs["history"]["lengths"])

            last_labels_embeddings = self.backbone.item_embeddings(last_labels)
            all_item_embeddings = self.backbone.item_embeddings.weight

            all_scores = last_embeddings @ all_item_embeddings.t()
            positive_score = (last_embeddings * last_labels_embeddings).sum(dim=-1)

            return {
                "all_scores": all_scores,
                "positive_scores": positive_score,
            }


class SASRecReal(SASRecModel):
    """
    SASRec model trained with Binary Cross-Entropy loss (negative sampling).
    """

    @classmethod
    def compute_loss_inner(
        cls,
        user_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        pos_logits = (user_embeddings * positive_embeddings).sum(dim=-1)
        neg_logits = (user_embeddings * negative_embeddings).sum(dim=-1)

        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))

        return (pos_loss + neg_loss) * 0.5

    def compute_loss(self, inputs: Dict, backbone_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        query_embeddings = backbone_output["encoder_output"]
        positive_embeddings = self.backbone.item_embeddings(inputs["labels"]["item_id"])

        random_item_ids = torch.randint(
            low=0,
            high=self.backbone.num_items,
            size=(positive_embeddings.shape[0],),
            device=positive_embeddings.device,
        )

        negative_embeddings = self.backbone.item_embeddings(random_item_ids)

        return self.compute_loss_inner(query_embeddings, positive_embeddings, negative_embeddings)


class SASRecInBatch(SASRecModel):
    """
    SASRec model trained with using in-batch negative sampling.
    """

    def __init__(self, backbone: SASRecBackbone, num_negatives: int) -> None:
        super().__init__(backbone)
        self.num_negatives = num_negatives

    @classmethod
    def compute_loss_inner(
        cls,
        user_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        pos_logits = (user_embeddings * positive_embeddings).sum(dim=-1)
        neg_logits = (user_embeddings.unsqueeze(1) * negative_embeddings).sum(dim=-1)
        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
        targets = torch.zeros(user_embeddings.shape[0], dtype=torch.long, device=user_embeddings.device)
        loss = F.cross_entropy(logits, targets)

        return loss

    def compute_loss(self, inputs: Dict[str, torch.Tensor], backbone_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        query_embeddings = backbone_output["encoder_output"]
        positive_embeddings = self.backbone.item_embeddings(inputs["labels"]["item_id"])

        inbatch_negative_ids = torch.randint(
            low=0,
            high=inputs["labels"]["item_id"].shape[0],
            size=(self.num_negatives * query_embeddings.shape[0],),
            device=positive_embeddings.device,
        )
        inbatch_item_ids = inputs["labels"]["item_id"][inbatch_negative_ids]

        inbatch_negative_embeddings = self.backbone.item_embeddings(inbatch_item_ids).reshape(
            query_embeddings.shape[0], self.num_negatives, query_embeddings.shape[-1]
        )

        return self.compute_loss_inner(query_embeddings, positive_embeddings, inbatch_negative_embeddings)


class SASRecInBatchWithLogQ(SASRecModel):
    """
    SASRec with in-batch negative sampling and LogQ bias correction.
    """

    def __init__(self, backbone: SASRecBackbone, num_negatives: int, item_freqs: torch.Tensor) -> None:
        super().__init__(backbone=backbone)
        self.num_negatives = num_negatives
        self.register_buffer("item_freqs", item_freqs)

    @staticmethod
    def apply_correction(scores: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        return scores - torch.log(freqs + 1e-9)

    @classmethod
    def compute_loss_inner(
        cls,
        user_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
        negative_item_ids: torch.Tensor,
        num_negatives: int,
        item_freqs: torch.Tensor,
    ) -> torch.Tensor:
        pos_logits = (user_embeddings * positive_embeddings).sum(dim=-1)  # (n,)
        neg_logits = (user_embeddings.unsqueeze(1) * negative_embeddings).sum(dim=-1)  # (n, k)

        neg_freqs = item_freqs[negative_item_ids].reshape(user_embeddings.shape[0], num_negatives)
        neg_logits = cls.apply_correction(neg_logits, neg_freqs)

        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)  # (n , k + 1)
        targets = torch.zeros(user_embeddings.shape[0], dtype=torch.long, device=user_embeddings.device)

        return F.cross_entropy(logits, targets)

    def compute_loss(self, inputs: Dict[str, torch.Tensor], backbone_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        query_embeddings = backbone_output["encoder_output"]
        positive_embeddings = self.backbone.item_embeddings(inputs["labels"]["item_id"])

        inbatch_negative_ids = torch.randint(
            low=0,
            high=inputs["labels"]["item_id"].shape[0],
            size=(self.num_negatives * query_embeddings.shape[0],),
            device=positive_embeddings.device,
        )
        inbatch_item_ids = inputs["labels"]["item_id"][inbatch_negative_ids]

        inbatch_negative_embeddings = self.backbone.item_embeddings(inbatch_item_ids).reshape(
            query_embeddings.shape[0], self.num_negatives, query_embeddings.shape[-1]
        )

        return self.compute_loss_inner(
            query_embeddings,
            positive_embeddings,
            inbatch_negative_embeddings,
            inbatch_item_ids,
            self.num_negatives,
            self.item_freqs,
        )
