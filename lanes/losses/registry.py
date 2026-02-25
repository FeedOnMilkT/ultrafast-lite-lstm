from lanes.losses.focal import SoftmaxFocalLoss
from lanes.losses.relation import ParsingRelationDis, ParsingRelationLoss


def build_losses():
    return {
        "cls_loss": SoftmaxFocalLoss(gamma=2.0),
        "relation_loss": ParsingRelationLoss(),
        "relation_dis": ParsingRelationDis(),
    }


def compute_total_loss(losses, logits, labels, sim_loss_w: float = 1.0, shp_loss_w: float = 0.0):
    cls_loss = losses["cls_loss"](logits, labels)
    relation_loss = losses["relation_loss"](logits)
    relation_dis = losses["relation_dis"](logits)
    total = cls_loss + sim_loss_w * relation_loss + shp_loss_w * relation_dis
    return total, {
        "cls_loss": float(cls_loss.detach().cpu()),
        "relation_loss": float(relation_loss.detach().cpu()),
        "relation_dis": float(relation_dis.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }
