from PIL import Image
import os
import pickle
import numpy as np
import torch
from io import BytesIO

_SCORER_CACHE = {}


def _get_cached_scorer(scorer_class, device, **kwargs):
    """Get or create a cached scorer instance to avoid duplicate model loading."""
    import inspect

    device_str = str(device)
    cache_key = (scorer_class.__name__, device_str)

    if cache_key not in _SCORER_CACHE:
        try:
            sig = inspect.signature(scorer_class.__init__)
            takes_device = "device" in sig.parameters
        except (ValueError, TypeError):
            takes_device = False

        if takes_device:
            _SCORER_CACHE[cache_key] = scorer_class(device=device, **kwargs)
        else:
            _SCORER_CACHE[cache_key] = scorer_class(**kwargs)

    return _SCORER_CACHE[cache_key]


def pickscore_photorealism(device):
    """PickScore with photorealistic prompt prefix for measuring photo-realism alignment."""
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = _get_cached_scorer(PickScoreScorer, device, dtype=torch.float32)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        pil_images = [Image.fromarray(image) for image in images]

        modified_prompts = ["A photorealistic, high quality, 4K, camera-captured snapshot of " + p for p in prompts]
        scores = scorer(modified_prompts, pil_images).cpu().numpy()
        return scores, {}

    return _fn


def _make_qwen_style_scorer(server_mode: str):
    """Factory: returns a reward function that scores images via the Qwen-VL reward server.

    Uses logit-based scoring (no text generation / parsing) for stability.
    The server_mode maps to a /mode/<mode> endpoint on the reward server
    (e.g. "t2i_style_logits").
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry

    base_url = os.getenv("QWEN_VL_REWARD_URL", "http://127.0.0.1:12341").strip().rstrip("/")
    timeout = float(os.getenv("QWEN_VL_REWARD_TIMEOUT", "1800"))
    session = requests.Session()
    retry = Retry(total=10, backoff_factor=2, status_forcelist=[500, 502, 503, 504], allowed_methods=False)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images_np = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images_np = images_np.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        else:
            images_np = images

        pil_images = [Image.fromarray(img) for img in images_np]

        image_bytes_list = []
        for img in pil_images:
            buf = BytesIO()
            img.save(buf, format="JPEG")
            image_bytes_list.append(buf.getvalue())

        payload = {"images": image_bytes_list, "prompts": list(prompts)}
        response = session.post(
            f"{base_url}/mode/{server_mode}",
            data=pickle.dumps(payload),
            timeout=timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Qwen style reward server error ({server_mode}): "
                f"status={response.status_code} body={response.content[:200]!r}"
            )
        result = pickle.loads(response.content)
        scores = result["scores"]
        return scores, {}

    return _fn


def qwen_style_sketch(device):
    return _make_qwen_style_scorer("t2i_style_logits")


MULTI_OUTPUT_SCORERS = {}


def expand_reward_fn_to_objectives(reward_fn):
    """Expand reward_fn (dict of score_name -> weight) to a list of (objective_name, weight).

    Single-output scorers yield one (name, weight). Multi-output scorers (listed in
    MULTI_OUTPUT_SCORERS) yield one (sub_name, weight) per sub-objective (same weight
    for each sub-objective).
    """
    expanded = []
    for name, weight in reward_fn.items():
        if name in MULTI_OUTPUT_SCORERS:
            for sub_name in MULTI_OUTPUT_SCORERS[name]:
                expanded.append((sub_name, weight))
        else:
            expanded.append((name, weight))
    return expanded


_SCORE_FUNCTIONS = {
    "pickscore_photorealism": pickscore_photorealism,
    "qwen_style_sketch": qwen_style_sketch,
}


def multi_score(device, score_dict):
    score_fns = {}
    for score_name, weight in score_dict.items():
        fn = _SCORE_FUNCTIONS[score_name]
        score_fns[score_name] = (
            fn(device) if "device" in fn.__code__.co_varnames else fn()
        )

    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        total_scores = []
        score_details = {}

        for score_name, weight in score_dict.items():
            scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            if isinstance(rewards, dict):
                for metric_name, metric_values in rewards.items():
                    if metric_name == "reasonings":
                        continue
                    detail_key = f"{score_name}_{metric_name}"
                    score_details[detail_key] = metric_values
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details["avg"] = total_scores
        return score_details, {}

    return _fn


def multi_score_conditioned(device, score_dict, constraints=None):
    score_fns = {}
    for score_name, weight in score_dict.items():
        fn = _SCORE_FUNCTIONS[score_name]
        score_fns[score_name] = (
            fn(device) if "device" in fn.__code__.co_varnames else fn()
        )

    def _fn(images, prompts, metadata, only_strict=True):
        raw_scores = {}
        extra_details = {}

        for score_name, weight in score_dict.items():
            scores, _ = score_fns[score_name](images, prompts, metadata)

            if not isinstance(scores, torch.Tensor):
                scores = torch.as_tensor(scores, device=device, dtype=torch.float32)
            raw_scores[score_name] = scores

        if constraints:
            for c in constraints:
                style_key = c["constrained"]
                guard_key = c["by"]
                lam = float(c.get("lambda", 5.0))
                eps = float(c.get("epsilon", 0.25))

                if style_key in raw_scores and guard_key in raw_scores:
                    penalty = lam * torch.clamp(eps - raw_scores[guard_key], min=0.0)
                    raw_scores[style_key] = raw_scores[style_key] - penalty

        weighted_scores_vector = {
            name: raw_scores[name] * float(score_dict[name])
            for name in raw_scores
        }

        score_details = dict(raw_scores)
        score_details.update(extra_details)
        score_details["weighted_scores"] = weighted_scores_vector

        return score_details, {}

    return _fn


def main():
    import torchvision.transforms as transforms

    image_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_cases/nasa.jpg"),
    ]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    images = torch.stack([transform(Image.open(image_path).convert("RGB")) for image_path in image_paths])
    prompts = [
        'An astronaut\'s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}
    score_dict = {"pickscore_photorealism": 1.0}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    scores, _ = scoring_fn(images, prompts, metadata)
    print("Scores:", scores)


if __name__ == "__main__":
    main()
