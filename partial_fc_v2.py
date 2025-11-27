from typing import Callable

import torch
from torch.nn.functional import linear, normalize, cross_entropy

class PartialFC_V2(torch.nn.Module):
    _version = 2

    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False,
    ):
        super(PartialFC_V2, self).__init__()

        self.rank = 0
        self.world_size = 1

        self.dist_cross_entropy = DistCrossEntropy()
        self.embedding_size = embedding_size
        self.sample_rate: float = float(sample_rate)
        self.fp16 = fp16

        self.num_local: int = int(num_classes)
        self.class_start: int = 0
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.last_batch_size: int = 0

        self.is_updated: bool = True
        self.init_weight_update: bool = True
        self.weight = torch.nn.Parameter(
            torch.normal(0, 0.01, (self.num_local, embedding_size))
        )

        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise ValueError("margin_loss must be a callable")

    def sample(self, labels, index_positive):
        with torch.no_grad():
            device = self.weight.device
            positive = torch.unique(labels[index_positive].view(-1), sorted=True).to(device)

            if self.num_sample - int(positive.size(0)) >= 0:
                perm = torch.rand(size=[self.num_local], device=device)
                perm[positive.long()] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1]
                index, _ = index.sort()
            else:
                index = positive.long().sort()[0]

            self.weight_index = index

            labels[index_positive] = torch.searchsorted(index, labels[index_positive].long())

        return self.weight[self.weight_index]

    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
    ):
        local_labels = local_labels.squeeze().long()
        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert (
            self.last_batch_size == batch_size
        ), f"last batch size do not equal current batch size: {self.last_batch_size} vs {batch_size}"

        embeddings = local_embeddings
        labels = local_labels.view(-1, 1)

        index_positive = (self.class_start <= labels) & (labels < self.class_start + self.num_local)
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1.0:
            weight = self.sample(labels, index_positive)
        else:
            weight = self.weight

        with torch.amp.autocast(device_type="cuda", enabled=self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1.0, 1.0)

        logits = self.margin_softmax(logits, labels)
        loss = self.dist_cross_entropy(logits, labels)
        return loss


class DistCrossEntropy(torch.nn.Module):

    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(self, logit_part: torch.Tensor, label_part: torch.Tensor):
        labels = label_part.view(-1).long()
        return cross_entropy(logit_part, labels, ignore_index=-1, reduction="mean")

def AllGather(tensor, *args):
    return [tensor]
