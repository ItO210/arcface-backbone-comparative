from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .mobilefacenet import get_mbf
from .shufflefacenet import shufflefacenet_05x, shufflefacenet_1x, shufflefacenet_15x, shufflefacenet_2x
from .mixfacenet import  MixFaceNet_XS, MixFaceNet_S, MixFaceNet_M, ShuffleMixFaceNet_XS, ShuffleMixFaceNet_S, ShuffleMixFaceNet_M

def get_model(name, **kwargs):

    # iResNet
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)

    # MobileFaceNet
    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf(fp16=fp16, num_features=num_features)

    # ShuffleFaceNet
    elif name == "sfn0_5":
        return shufflefacenet_05x(**kwargs)
    elif name == "sfn1_5":
        return shufflefacenet_1x(**kwargs)
    elif name == "sfn1_0":
        return shufflefacenet_15x(**kwargs)
    elif name == "sfn2_0":
        return shufflefacenet_2x(**kwargs)

    # MixFaceNet
    elif name == "mfn_xs":
        fp16 = kwargs.get("fp16", False)
        return MixFaceNet_XS(fp16=fp16)
    elif name == "mfn_s":
        fp16 = kwargs.get("fp16", False)
        return MixFaceNet_S(fp16=fp16)
    elif name == "mfn_m":
        fp16 = kwargs.get("fp16", False)
        return MixFaceNet_M(fp16=fp16)

    # ShuffleMixFaceNet family
    elif name == "smfn_xs":
        fp16 = kwargs.get("fp16", False)
        return ShuffleMixFaceNet_XS(fp16=fp16)
    elif name == "smfn_s":
        fp16 = kwargs.get("fp16", False)
        return ShuffleMixFaceNet_S(fp16=fp16)
    elif name == "smfn_m":
        fp16 = kwargs.get("fp16", False)
        return ShuffleMixFaceNet_M(fp16=fp16)

    else:
        raise ValueError(f"Unknown model name: {name}")
