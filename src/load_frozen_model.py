import torch
from models_archs import TransformerNoduleBimodalClassifier, TransformerNoduleClassifier


def build_frozen_model(
    cfg, arch, modality, modality_a, modality_b, feature_dim, num_classes=2
):
    cfg_model = cfg["models"][arch]

    if modality == "petct" or modality == "petchest":
        print("\nUsing Bimodal Classifier\n")
        mlp_ratio_ct = cfg_model[modality_b]["mlp_ratio"]
        mlp_ratio_pet = cfg_model[modality_a]["mlp_ratio"]

        num_heads_ct = cfg_model[modality_b]["num_heads"]
        num_heads_pet = cfg_model[modality_a]["num_heads"]

        num_layers_ct = cfg_model[modality_b]["num_layers"]
        num_layers_pet = cfg_model[modality_a]["num_layers"]

        model = TransformerNoduleBimodalClassifier(
            feature_dim,
            mlp_ratio_ct,
            mlp_ratio_pet,
            num_heads_ct,
            num_heads_pet,
            num_layers_ct,
            num_layers_pet,
            True,
            num_classes=num_classes,
        )
    else:
        print("\nUsing Monomodal Classifier\n")

        mlp_ratio = cfg_model[modality]["mlp_ratio"]
        num_heads = cfg_model[modality]["num_heads"]
        num_layers = cfg_model[modality]["num_layers"]
        dim_feedforward = int(feature_dim * mlp_ratio)
        model = TransformerNoduleClassifier(
            input_dim=feature_dim,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            num_classes=num_classes,
            num_layers=num_layers,
        )
    return model


def add_mode(param, mode):
    words = param.split(".")
    words[0] = words[0] + f"_{mode}"
    new_name = ".".join(words)
    return new_name


def load_frozen_transformers(
    cfg,
    model_path,
    feature_dim,
    arch,
    modality,
    modality_a,
    modality_b,
    num_classes=2,
):
    model = build_frozen_model(
        cfg, arch, modality, modality_a, modality_b, feature_dim, num_classes
    )

    if modality == "petct":
        params = {}
        model_weights = torch.load(model_path, map_location="cpu", weights_only=True)

        important_keys = ("cls_token", "transformer_encoder", "norm")
        for k, v in model_weights.items():
            if any(important_key in k for important_key in important_keys):
                params[k] = v
    elif modality == "pet":
        pet_weight = torch.load(model_path, map_location="cpu", weights_only=True)
        params= {
            add_mode(k, "pet"): v
            for k, v in pet_weight.items()
            if k.startswith(("cls_token", "transformer_encoder", "norm"))
        }

    model.load_state_dict(params, strict=False)
    print("\n\n Loaded Transformers weights \n\n")
    return model
