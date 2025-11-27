# Run this to get GFLOPs and number of parameters.

from ptflops import get_model_complexity_info
from backbones import get_model

def measure_model(model_name):
    net = get_model(model_name).eval()

    macs, params = get_model_complexity_info(
        net,
        (3, 112, 112),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )

    true_gflops = (macs * 2) / 1e9
    mparams = params / 1e6

    return model_name, true_gflops, mparams


def measure_models(model_list):
    results = [measure_model(m) for m in model_list]

    header = f"{'Model':<15} {'GFLOPs':<10} {'Params (M)':<12}"
    print(header)
    print("-" * len(header))

    for model, flops, params in results:
        print(f"{model:<15} {flops:<10.3f} {params:<12.3f}")

models_to_test = [
    "mbf",
    "mfn_m",
    "mfn_s",
    "mfn_xs",
    "r100",
    "r50",
    "r18",
    "sfn0_5",
    "sfn1_0",
    "sfn1_5",
    "sfn2_0",
    "smfn_m",
    "smfn_s",
    "smfn_xs",
]
measure_models(models_to_test)
