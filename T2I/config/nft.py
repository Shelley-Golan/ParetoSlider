import importlib.util
import os

_spec = importlib.util.spec_from_file_location("base", os.path.join(os.path.dirname(__file__), "base.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def get_config(name):
    return globals()[name]()


def _get_config(base_model="sd3", n_gpus=1, gradient_step_per_epoch=1, dataset="pickscore", reward_fn={}, name=""):
    config = base.get_config()
    assert base_model in ["sd3"]
    assert dataset in ["pickscore", "ocr", "geneval", "sharegpt"]

    config.base_model = base_model
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    if base_model == "sd3":
        config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
        config.sample.num_steps = 5
        config.sample.eval_num_steps = 40
        config.sample.guidance_scale = 4.5
        config.resolution = 512
        config.train.beta = 0.0001
        config.sample.noise_level = 0.7
        bsz = 12

    config.sample.num_image_per_prompt = 24
    num_groups = 48

    while True:
        if bsz < 1:
            assert False, "Cannot find a proper batch size."
        if (
            num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_image_per_prompt == 0
        ):
            n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
            if n_batch_per_epoch % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = bsz
                config.sample.num_batches_per_epoch = n_batch_per_epoch
                config.train.batch_size = config.sample.train_batch_size
                config.train.gradient_accumulation_steps = (
                    config.sample.num_batches_per_epoch // gradient_step_per_epoch
                )
                break
        bsz -= 1

    config.sample.test_batch_size = 14 if dataset == "geneval" else 16
    if n_gpus > 32:
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    if dataset == "geneval":
        config.prompt_fn = "geneval"
    elif dataset == "sharegpt":
        config.prompt_fn = "sharegpt"
    else:
        config.prompt_fn = "general_ocr"

    config.run_name = f"nft_{base_model}_{name}"
    config.save_dir = os.path.join(config.logdir, "nft", base_model, name)
    config.reward_fn = reward_fn
    config.reward_constraints = []

    config.decay_type = 1
    config.beta = 1.0
    config.train.adv_mode = "all"

    config.sample.guidance_scale = 1.0
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    return config


def _sd3_style_config(style_key, style_short_name):
    """Shared config for pickscore_photorealism + a style objective."""
    reward_fn = {
        "pickscore_photorealism": 1.0,
        style_key: 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=4,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name=f"pickscore_photorealism_{style_short_name}_init51",
    )
    config.sample.num_steps = 25
    config.beta = 0.1
    config.train.beta = 0.0001
    config.loss_mode = "per_objective"
    config.num_pref_per_prompt = 3
    config.block_mod_form = "residual"
    config.mod_block_fraction = 1.0
    config.num_epochs = 31
    config.num_freqs = 1
    config.train.lora_rank = 32
    config.train.lora_path = os.environ.get("NFT_LORA_PATH", None)
    return config


def sd3_qwen_style_sketch():
    return _sd3_style_config("qwen_style_sketch", "qwen_sketch")


def sd3_qwen_sketch_photorealism_single_loss():
    """2 objectives with single (weighted-sum) loss."""
    reward_fn = {
        "pickscore_photorealism": 1.0,
        "qwen_style_sketch": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=6,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="qwen_sketch_photorealism_init51_single_loss",
    )
    config.sample.num_steps = 25
    config.beta = 0.1
    config.train.beta = 0.01
    config.loss_mode = "single_loss"
    config.num_pref_per_prompt = 3
    config.block_mod_form = "residual"
    config.mod_block_fraction = 1.0
    config.num_epochs = 31
    config.num_freqs = 1
    config.train.lora_rank = 32
    config.train.lora_path = os.environ.get("NFT_LORA_PATH", None)
    return config


def sd3_1_pickscore_photorealism_0_qwen_style_sketch():
    """Single-objective baseline: photorealism only (sketch weight = 0)."""
    reward_fn = {
        "pickscore_photorealism": 1.0,
        "qwen_style_sketch": 0.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=6,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="pickscore_photorealism_1_qwen_style_sketch_0_init51",
    )
    config.sample.num_steps = 25
    config.beta = 0.1
    config.train.beta = 0.0001
    config.train.lora_path = os.environ.get("NFT_LORA_PATH", None)
    return config
