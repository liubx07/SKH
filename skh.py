import torch
from torch.nn.functional import linear, normalize, cross_entropy
from partial_fc import CosFace, ArcFace

class NaiveFCSKH(torch.nn.Module):
    def __init__(
            self,
            embedding_size,
            num_classes,
            skh_k=3,
            fp16: bool = False,
            margin_loss = "cosface",
            s=50, m=0.3,
    ):
        super().__init__()
        self.fp16 = fp16
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes*skh_k, embedding_size)))
        self.skh_k = skh_k
        if isinstance(margin_loss, str):
            self.margin_softmax: torch.nn.Module
            if margin_loss == "cosface":
                self.margin_softmax = CosFace(s=s, m=m)
            elif margin_loss == "arcface":
                self.margin_softmax = ArcFace(s=s, m=m)
            else:
                raise
        else:
            raise

    def forward(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor,
    ):
        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight = normalize(self.weight)
            logits = linear(norm_embeddings, norm_weight)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        b = embeddings.shape[0]
        logits_k = logits.reshape(b, self.skh_k, -1)
        loss_list = []
        for i in range(self.skh_k):
            logits = self.margin_softmax(logits_k[:,i], labels)
            loss = cross_entropy(logits, labels, reduction='none')
            loss_list.append(loss[:,None])
        loss_list = torch.cat(loss_list, dim=1)
        loss = torch.min(loss_list, dim=1)[0].mean()
        return loss