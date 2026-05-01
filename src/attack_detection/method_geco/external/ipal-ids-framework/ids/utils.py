from .GeCo.GeCo import GeCo
from .Knowledgebased.ExpertInvariants import ExpertInvariants

idss = [
    GeCo,
    ExpertInvariants,
]


def get_all_iidss():
    return {ids._name: ids for ids in idss}
