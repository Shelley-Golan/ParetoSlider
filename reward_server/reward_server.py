import json
import re
from typing import List
from vllm import LLM, SamplingParams
from io import BytesIO
import base64
import pickle
import traceback
from flask import Flask, request
import ray
import asyncio
import prompt_template


app = Flask(__name__)

# Global variables
workers = []  # Ray actors for each GPU
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
NUM_GPUS = 1
NUM_TP = 1

def get_base64(image):
    image_data = BytesIO()
    image.save(image_data, format="JPEG")
    image_data_bytes = image_data.getvalue()
    encoded_image = base64.b64encode(image_data_bytes).decode("utf-8")
    return encoded_image



def parse_unified_reward_json(text):
    """Parse JSON output from UnifiedReward-Edit model.

    Expected: {"reasoning": "...", "score": [score1, score2]}
    Returns (score1, score2) normalized to [0, 1] from [0, 25] scale.
    Raises ValueError on parse failure (no silent fallbacks).
    """
    # Try direct JSON parse
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Model sometimes wraps JSON in extra text; try to extract it
        match = re.search(r'\{[^{}]*"score"\s*:\s*\[.*?\][^{}]*\}', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if parsed is None:
        # Last resort: extract score list directly via regex
        match = re.search(r'"score"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', text)
        if match:
            s1 = max(0.0, min(25.0, float(match.group(1)))) / 25.0
            s2 = max(0.0, min(25.0, float(match.group(2)))) / 25.0
            return (s1, s2)
        raise ValueError(f"Failed to parse UnifiedReward JSON: {text[:500]}")

    scores = parsed.get("score", [])
    if not isinstance(scores, list) or len(scores) < 2:
        raise ValueError(f"Unexpected score format (need list of 2+): {text[:500]}")

    s1 = max(0.0, min(25.0, float(scores[0]))) / 25.0
    s2 = max(0.0, min(25.0, float(scores[1]))) / 25.0
    return (s1, s2)


@ray.remote(num_gpus=NUM_TP)
class ModelWorker:
    def __init__(self):
        self.llm = None
        self.load_model()

    def load_model(self):
        """Load the Qwen2-VL model using vLLM on specific GPU"""
        self.llm = LLM(
            MODEL_PATH, limit_mm_per_prompt={"image": 3}, tensor_parallel_size=NUM_TP,
            trust_remote_code=True,
        )
        # Compute token IDs dynamically from tokenizer
        tokenizer = self.llm.get_tokenizer()

        # 0-5 scale (default scoring)
        self.score_idx_5 = []
        self.score_values_5 = []
        for i in range(6):
            toks = tokenizer.encode(str(i), add_special_tokens=False)
            if len(toks) == 1:
                self.score_idx_5.append(toks[0])
                self.score_values_5.append(i)
        print(f"0-5 score tokens: {self.score_values_5} (token IDs: {self.score_idx_5})")

        # 0-9 scale (edit success/preservation scoring)
        self.score_idx_9 = []
        self.score_values_9 = []
        for i in range(10):
            toks = tokenizer.encode(str(i), add_special_tokens=False)
            if len(toks) == 1:
                self.score_idx_9.append(toks[0])
                self.score_values_9.append(i)
        print(f"0-9 score tokens: {self.score_values_9} (token IDs: {self.score_idx_9})")

    def evaluate_image(
        self, image_bytes, prompt, ref_image_bytes=None, requirement: str = ""
    ):
        conversation = [
            {
                "role": "user",
                "content": [
                    self._bytes_to_image_url(ref_image_bytes),
                    self._bytes_to_image_url(image_bytes),
                    {
                        "type": "text",
                        "text": prompt_template.SCORE_LOGIT.format(
                            prompt=prompt, requirement=requirement
                        ),
                    },
                ],
            },
        ]
        return self._vllm_evaluate(conversation)

    def evaluate_edit_image(self, image_bytes, prompt, ref_image_bytes, editing_task=""):
        """Evaluate an edited image against the original using the edit instruction."""
        conversation = [
            {
                "role": "user",
                "content": [
                    self._bytes_to_image_url(ref_image_bytes),
                    self._bytes_to_image_url(image_bytes),
                    {
                        "type": "text",
                        "text": prompt_template.SCORE_EDIT_LOGIT.format(prompt=prompt, editing_task=editing_task),
                    },
                ],
            },
        ]
        return self._vllm_evaluate(conversation)

    def evaluate_edit_success_image(self, image_bytes, prompt, ref_image_bytes, editing_task=""):
        """Evaluate editing instruction following (success) on 0-9 scale via logits."""
        conversation = [
            {
                "role": "user",
                "content": [
                    self._bytes_to_image_url(ref_image_bytes),
                    self._bytes_to_image_url(image_bytes),
                    {
                        "type": "text",
                        "text": prompt_template.SCORE_EDIT_SUCCESS_LOGIT.format(prompt=prompt, editing_task=editing_task),
                    },
                ],
            },
        ]
        return self._vllm_evaluate(conversation, custom_score_idx=self.score_idx_9, custom_score_values=self.score_values_9)

    def evaluate_edit_preservation_image(self, image_bytes, prompt, ref_image_bytes, editing_task=""):
        """Evaluate preservation of unrelated details on 0-9 scale via logits."""
        conversation = [
            {
                "role": "user",
                "content": [
                    self._bytes_to_image_url(ref_image_bytes),
                    self._bytes_to_image_url(image_bytes),
                    {
                        "type": "text",
                        "text": prompt_template.SCORE_EDIT_PRESERVATION_LOGIT.format(prompt=prompt, editing_task=editing_task),
                    },
                ],
            },
        ]
        return self._vllm_evaluate(conversation, custom_score_idx=self.score_idx_9, custom_score_values=self.score_values_9)

    def evaluate_t2i_image(self, image_bytes, prompt):
        """Evaluate a text-to-image generated image (no reference image needed)."""
        conversation = [
            {
                "role": "user",
                "content": [
                    self._bytes_to_image_url(image_bytes),
                    {"type": "text", "text": prompt_template.SCORE_T2I_LOGIT.format(prompt=prompt)},
                ],
            },
        ]
        return self._vllm_evaluate(conversation)

    def evaluate_t2i_style_image(self, image_bytes, prompt):
        """Evaluate T2I image for sketch/style adherence."""
        conversation = [
            {
                "role": "user",
                "content": [
                    self._bytes_to_image_url(image_bytes),
                    {"type": "text", "text": prompt_template.SCORE_T2I_LOGIT_SKETCH.format(prompt=prompt)},
                ],
            },
        ]
        return self._vllm_evaluate(conversation)

    def evaluate_t2i_van_gogh_image(self, image_bytes, prompt):
        """Evaluate T2I image for Van Gogh style adherence."""
        conversation = [
            {
                "role": "user",
                "content": [
                    self._bytes_to_image_url(image_bytes),
                    {"type": "text", "text": prompt_template.SCORE_T2I_LOGIT_VAN_GOGH.format(prompt=prompt)},
                ],
            },
        ]
        return self._vllm_evaluate(conversation)

    def evaluate_t2i_vector_art_image(self, image_bytes, prompt):
        """Evaluate T2I image for vector art style adherence."""
        conversation = [
            {
                "role": "user",
                "content": [
                    self._bytes_to_image_url(image_bytes),
                    {"type": "text", "text": prompt_template.SCORE_T2I_LOGIT_VECTOR_ART.format(prompt=prompt)},
                ],
            },
        ]
        return self._vllm_evaluate(conversation)

    def evaluate_t2i_pixel_art_image(self, image_bytes, prompt):
        """Evaluate T2I image for pixel art style adherence."""
        conversation = [
            {
                "role": "user",
                "content": [
                    self._bytes_to_image_url(image_bytes),
                    {"type": "text", "text": prompt_template.SCORE_T2I_LOGIT_PIXEL_ART.format(prompt=prompt)},
                ],
            },
        ]
        return self._vllm_evaluate(conversation)

    def evaluate_t2i_photorealism_image(self, image_bytes, prompt):
        """Evaluate T2I image for photorealism quality."""
        conversation = [
            {
                "role": "user",
                "content": [
                    self._bytes_to_image_url(image_bytes),
                    {"type": "text", "text": prompt_template.SCORE_T2I_LOGIT_PHOTOREALISM.format(prompt=prompt)},
                ],
            },
        ]
        return self._vllm_evaluate(conversation)

    def evaluate_t2i_watercolor_image(self, image_bytes, prompt):
        """Evaluate T2I image for watercolor style adherence."""
        conversation = [
            {
                "role": "user",
                "content": [
                    self._bytes_to_image_url(image_bytes),
                    {"type": "text", "text": prompt_template.SCORE_T2I_LOGIT_WATERCOLOR.format(prompt=prompt)},
                ],
            },
        ]
        return self._vllm_evaluate(conversation)

    def evaluate_t2i_animation_image(self, image_bytes, prompt):
        """Evaluate T2I image for animation/cartoon style adherence."""
        conversation = [
            {
                "role": "user",
                "content": [
                    self._bytes_to_image_url(image_bytes),
                    {"type": "text", "text": prompt_template.SCORE_T2I_LOGIT_ANIMATION.format(prompt=prompt)},
                ],
            },
        ]
        return self._vllm_evaluate(conversation)

    def evaluate_t2i_generic_style_image(self, image_bytes, prompt, template_attr):
        """Evaluate T2I image using any SCORE_T2I_LOGIT_* prompt template by attribute name."""
        template = getattr(prompt_template, template_attr)
        conversation = [
            {
                "role": "user",
                "content": [
                    self._bytes_to_image_url(image_bytes),
                    {"type": "text", "text": template.format(prompt=prompt)},
                ],
            },
        ]
        return self._vllm_evaluate(conversation)

    def _vllm_generate(self, conversation, max_tokens=4096, temperature=0.7):
        """Generate text from the model and return the raw output string."""
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        outputs = self.llm.chat(conversation, sampling_params=sampling_params)
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text
        return ""

    def _bytes_to_image_url(self, jpeg_bytes):
        """Convert JPEG bytes to base64 image_url dict for OpenAI chat format."""
        b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

    def evaluate_edit_unified(self, image_bytes, prompt, ref_image_bytes):
        """Evaluate editing using UnifiedReward-Edit pointwise prompt.

        Uses temperature > 0 so that different samples for the same prompt
        get naturally varied scores across the K samples per prompt.
        Returns (edit_success_score, preservation_score) both in [0, 1].
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    self._bytes_to_image_url(ref_image_bytes),
                    self._bytes_to_image_url(image_bytes),
                    {
                        "type": "text",
                        "text": prompt_template.UNIFIED_REWARD_EDIT_POINTWISE.format(prompt=prompt),
                    },
                ],
            },
        ]
        text_output = self._vllm_generate(conversation, temperature=0.99)
        print(f"UnifiedReward-Edit raw output: {text_output}")
        scores = parse_unified_reward_json(text_output)
        print(f"UnifiedReward-Edit scores: success={scores[0]:.4f}, preservation={scores[1]:.4f} | prompt: {prompt}")
        return scores

    def _vllm_evaluate(self, conversation, max_tokens=1, max_score=5, custom_score_idx=None, custom_score_values=None):
        import math
        sampling_params = SamplingParams(
            max_tokens=max_tokens, logprobs=20, temperature=0,
        )
        outputs = self.llm.chat(conversation, sampling_params=sampling_params)
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("_vllm_evaluate: no output from model")

        output = outputs[0].outputs[0]
        if not output.logprobs or len(output.logprobs) == 0:
            raise RuntimeError("_vllm_evaluate: no logprobs in output")

        first_token_logprobs = output.logprobs[0]  # dict: token_id -> Logprob

        _idx = custom_score_idx if custom_score_idx is not None else self.score_idx_5
        if custom_score_values is not None:
            _values = custom_score_values
            _max = max(custom_score_values)
        else:
            _values = self.score_values_5
            _max = max(self.score_values_5)

        # Extract probabilities for score tokens from top-K logprobs
        score_probs = []
        for token_id in _idx:
            if token_id in first_token_logprobs:
                score_probs.append(math.exp(first_token_logprobs[token_id].logprob))
            else:
                score_probs.append(0.0)

        # Normalize over score tokens only
        total = sum(score_probs)
        if total == 0:
            # Log what the model actually produced
            top_tokens = {tid: lp.decoded_token for tid, lp in first_token_logprobs.items()}
            raise RuntimeError(
                f"_vllm_evaluate: no score tokens found in top-20 logprobs. "
                f"Model produced: {top_tokens}"
            )
        score_probs = [p / total for p in score_probs]

        score = sum(p * v for p, v in zip(score_probs, _values)) / _max
        print(f"Score: {score:.4f}")
        return score


def initialize_ray_workers(num_gpus=8, num_tp=4):
    global workers
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(runtime_env={"excludes": [".venv/", "__pycache__/", ".git/"]})

    # Create workers for each GPU
    workers = []
    for _ in range(num_gpus // num_tp):
        worker = ModelWorker.remote()
        workers.append(worker)

    print(f"Initialized {num_gpus//num_tp} Ray workers")
    return workers


async def evaluate_images_async(
    image_bytes_list, prompts, ref_image_bytes_list=None, requirements: List[str] = []
):
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    tasks = []
    if not requirements:
        requirements = [""] * len(prompts)
    if ref_image_bytes_list is None:
        ref_image_bytes_list = [None] * len(prompts)
    for i, (image_bytes, prompt, ref_image_bytes, requirement) in enumerate(
        zip(image_bytes_list, prompts, ref_image_bytes_list, requirements)
    ):
        worker_idx = i % len(workers)
        worker = workers[worker_idx]
        task = worker.evaluate_image.remote(
            image_bytes, prompt, ref_image_bytes, requirement
        )
        tasks.append(task)

    scores = ray.get(tasks)
    return scores


def evaluate_images(
    image_bytes_list, prompts, ref_image_bytes_list=None, requirements=[]
):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            evaluate_images_async(
                image_bytes_list, prompts, ref_image_bytes_list, requirements
            )
        )
        return scores
    finally:
        loop.close()


async def evaluate_t2i_images_async(image_bytes_list, prompts):
    """Evaluate text-to-image generated images (no reference images)."""
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    tasks = []
    for i, (image_bytes, prompt) in enumerate(zip(image_bytes_list, prompts)):
        worker_idx = i % len(workers)
        worker = workers[worker_idx]
        task = worker.evaluate_t2i_image.remote(image_bytes, prompt)
        tasks.append(task)

    scores = ray.get(tasks)
    return scores


def evaluate_t2i_images(image_bytes_list, prompts):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            evaluate_t2i_images_async(image_bytes_list, prompts)
        )
        return scores
    finally:
        loop.close()


async def evaluate_t2i_style_images_async(image_bytes_list, prompts):
    """Evaluate T2I images for sketch/style adherence (no reference images)."""
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    tasks = []
    for i, (image_bytes, prompt) in enumerate(zip(image_bytes_list, prompts)):
        worker_idx = i % len(workers)
        worker = workers[worker_idx]
        task = worker.evaluate_t2i_style_image.remote(image_bytes, prompt)
        tasks.append(task)

    scores = ray.get(tasks)
    return scores


def evaluate_t2i_style_images(image_bytes_list, prompts):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            evaluate_t2i_style_images_async(image_bytes_list, prompts)
        )
        return scores
    finally:
        loop.close()


async def evaluate_t2i_pixel_art_images_async(image_bytes_list, prompts):
    """Evaluate T2I images for pixel art style adherence (no reference images)."""
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    tasks = []
    for i, (image_bytes, prompt) in enumerate(zip(image_bytes_list, prompts)):
        worker_idx = i % len(workers)
        worker = workers[worker_idx]
        task = worker.evaluate_t2i_pixel_art_image.remote(image_bytes, prompt)
        tasks.append(task)

    scores = ray.get(tasks)
    return scores


def evaluate_t2i_pixel_art_images(image_bytes_list, prompts):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            evaluate_t2i_pixel_art_images_async(image_bytes_list, prompts)
        )
        return scores
    finally:
        loop.close()


async def evaluate_t2i_vector_art_images_async(image_bytes_list, prompts):
    """Evaluate T2I images for vector art style adherence (no reference images)."""
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    tasks = []
    for i, (image_bytes, prompt) in enumerate(zip(image_bytes_list, prompts)):
        worker_idx = i % len(workers)
        worker = workers[worker_idx]
        task = worker.evaluate_t2i_vector_art_image.remote(image_bytes, prompt)
        tasks.append(task)

    scores = ray.get(tasks)
    return scores


def evaluate_t2i_vector_art_images(image_bytes_list, prompts):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            evaluate_t2i_vector_art_images_async(image_bytes_list, prompts)
        )
        return scores
    finally:
        loop.close()


async def evaluate_t2i_van_gogh_images_async(image_bytes_list, prompts):
    """Evaluate T2I images for Van Gogh style adherence (no reference images)."""
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    tasks = []
    for i, (image_bytes, prompt) in enumerate(zip(image_bytes_list, prompts)):
        worker_idx = i % len(workers)
        worker = workers[worker_idx]
        task = worker.evaluate_t2i_van_gogh_image.remote(image_bytes, prompt)
        tasks.append(task)

    scores = ray.get(tasks)
    return scores


def evaluate_t2i_van_gogh_images(image_bytes_list, prompts):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            evaluate_t2i_van_gogh_images_async(image_bytes_list, prompts)
        )
        return scores
    finally:
        loop.close()


async def evaluate_t2i_photorealism_images_async(image_bytes_list, prompts):
    """Evaluate T2I images for photorealism quality (no reference images)."""
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    tasks = []
    for i, (image_bytes, prompt) in enumerate(zip(image_bytes_list, prompts)):
        worker_idx = i % len(workers)
        worker = workers[worker_idx]
        task = worker.evaluate_t2i_photorealism_image.remote(image_bytes, prompt)
        tasks.append(task)

    scores = ray.get(tasks)
    return scores


def evaluate_t2i_photorealism_images(image_bytes_list, prompts):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            evaluate_t2i_photorealism_images_async(image_bytes_list, prompts)
        )
        return scores
    finally:
        loop.close()


async def evaluate_t2i_watercolor_images_async(image_bytes_list, prompts):
    """Evaluate T2I images for watercolor style adherence (no reference images)."""
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    tasks = []
    for i, (image_bytes, prompt) in enumerate(zip(image_bytes_list, prompts)):
        worker_idx = i % len(workers)
        worker = workers[worker_idx]
        task = worker.evaluate_t2i_watercolor_image.remote(image_bytes, prompt)
        tasks.append(task)

    scores = ray.get(tasks)
    return scores

def evaluate_t2i_watercolor_images(image_bytes_list, prompts):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            evaluate_t2i_watercolor_images_async(image_bytes_list, prompts)
        )
        return scores
    finally:
        loop.close()


async def evaluate_t2i_animation_images_async(image_bytes_list, prompts):
    """Evaluate T2I images for animation/cartoon style adherence (no reference images)."""
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    tasks = []
    for i, (image_bytes, prompt) in enumerate(zip(image_bytes_list, prompts)):
        worker_idx = i % len(workers)
        worker = workers[worker_idx]
        task = worker.evaluate_t2i_animation_image.remote(image_bytes, prompt)
        tasks.append(task)

    scores = ray.get(tasks)
    return scores


def evaluate_t2i_animation_images(image_bytes_list, prompts):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            evaluate_t2i_animation_images_async(image_bytes_list, prompts)
        )
        return scores
    finally:
        loop.close()


# --- Generic template-driven T2I style evaluation ---
# Maps server mode names to prompt_template attribute names
_T2I_STYLE_TEMPLATES = {
    "t2i_oil_painting_logits": "SCORE_T2I_LOGIT_OIL_PAINTING",
    "t2i_anime_logits": "SCORE_T2I_LOGIT_ANIME",
    "t2i_flat_vector_logits": "SCORE_T2I_LOGIT_FLAT_VECTOR",
    "t2i_vintage_film_logits": "SCORE_T2I_LOGIT_VINTAGE_FILM",
    "t2i_neon_cyberpunk_logits": "SCORE_T2I_LOGIT_NEON_CYBERPUNK",
    "t2i_low_poly_logits": "SCORE_T2I_LOGIT_LOW_POLY",
    "t2i_ukiyoe_logits": "SCORE_T2I_LOGIT_UKIYOE",
    "t2i_pastel_logits": "SCORE_T2I_LOGIT_PASTEL",
    "t2i_disney_logits": "SCORE_T2I_LOGIT_DISNEY",
    "t2i_comic_book_logits": "SCORE_T2I_LOGIT_COMIC_BOOK",
    "t2i_caricature_logits": "SCORE_T2I_LOGIT_CARICATURE",
    "t2i_origami_logits": "SCORE_T2I_LOGIT_ORIGAMI",
}


async def evaluate_t2i_generic_style_images_async(image_bytes_list, prompts, template_attr):
    """Evaluate T2I images using a generic style template (no reference images)."""
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    tasks = []
    for i, (image_bytes, prompt) in enumerate(zip(image_bytes_list, prompts)):
        worker_idx = i % len(workers)
        worker = workers[worker_idx]
        task = worker.evaluate_t2i_generic_style_image.remote(image_bytes, prompt, template_attr)
        tasks.append(task)

    scores = ray.get(tasks)
    return scores


def evaluate_t2i_generic_style_images(image_bytes_list, prompts, template_attr):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            evaluate_t2i_generic_style_images_async(image_bytes_list, prompts, template_attr)
        )
        return scores
    finally:
        loop.close()


async def evaluate_edit_images_async(image_bytes_list, prompts, ref_image_bytes_list, task_types):
    """Evaluate edited images against originals using the edit instruction."""
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    tasks = []
    for i, (image_bytes, prompt, ref_image_bytes, task_type) in enumerate(
        zip(image_bytes_list, prompts, ref_image_bytes_list, task_types)
    ):
        worker_idx = i % len(workers)
        worker = workers[worker_idx]
        task = worker.evaluate_edit_image.remote(image_bytes, prompt, ref_image_bytes, task_type)
        tasks.append(task)

    scores = ray.get(tasks)
    return scores


def evaluate_edit_images(image_bytes_list, prompts, ref_image_bytes_list, task_types):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            evaluate_edit_images_async(image_bytes_list, prompts, ref_image_bytes_list, task_types)
        )
        return scores
    finally:
        loop.close()


async def evaluate_edit_success_images_async(image_bytes_list, prompts, ref_image_bytes_list, task_types):
    """Evaluate editing success (instruction following) via logits on 0-25 scale."""
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    tasks = []
    for i, (image_bytes, prompt, ref_image_bytes, task_type) in enumerate(
        zip(image_bytes_list, prompts, ref_image_bytes_list, task_types)
    ):
        worker_idx = i % len(workers)
        worker = workers[worker_idx]
        task = worker.evaluate_edit_success_image.remote(image_bytes, prompt, ref_image_bytes, task_type)
        tasks.append(task)

    scores = ray.get(tasks)
    return scores


def evaluate_edit_success_images(image_bytes_list, prompts, ref_image_bytes_list, task_types):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            evaluate_edit_success_images_async(image_bytes_list, prompts, ref_image_bytes_list, task_types)
        )
        return scores
    finally:
        loop.close()


async def evaluate_edit_preservation_images_async(image_bytes_list, prompts, ref_image_bytes_list, task_types):
    """Evaluate preservation of unrelated details via logits on 0-25 scale."""
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    tasks = []
    for i, (image_bytes, prompt, ref_image_bytes, task_type) in enumerate(
        zip(image_bytes_list, prompts, ref_image_bytes_list, task_types)
    ):
        worker_idx = i % len(workers)
        worker = workers[worker_idx]
        task = worker.evaluate_edit_preservation_image.remote(image_bytes, prompt, ref_image_bytes, task_type)
        tasks.append(task)

    scores = ray.get(tasks)
    return scores


def evaluate_edit_preservation_images(image_bytes_list, prompts, ref_image_bytes_list, task_types):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            evaluate_edit_preservation_images_async(image_bytes_list, prompts, ref_image_bytes_list, task_types)
        )
        return scores
    finally:
        loop.close()


async def evaluate_edit_unified_async(image_bytes_list, prompts, ref_image_bytes_list):
    """Unified edit evaluation via UnifiedReward-Edit: returns (success_scores, preservation_scores)."""
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    tasks = []
    for i, (image_bytes, prompt, ref_image_bytes) in enumerate(
        zip(image_bytes_list, prompts, ref_image_bytes_list)
    ):
        worker_idx = i % len(workers)
        worker = workers[worker_idx]
        task = worker.evaluate_edit_unified.remote(image_bytes, prompt, ref_image_bytes)
        tasks.append(task)

    results = ray.get(tasks)  # list of (success, preservation) tuples
    success_scores = [r[0] for r in results]
    preservation_scores = [r[1] for r in results]
    return success_scores, preservation_scores


def evaluate_edit_unified_images(image_bytes_list, prompts, ref_image_bytes_list):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            evaluate_edit_unified_async(image_bytes_list, prompts, ref_image_bytes_list)
        )
    finally:
        loop.close()


@app.route("/mode/<mode>", methods=["POST"])
def inference_mode(mode):
    data = request.get_data()

    _valid_modes = [
        "logits_non_cot", "edit_logits", "edit_success_logits", "edit_preservation_logits",
        "edit_unified", "t2i_logits", "t2i_style_logits", "t2i_photorealism_logits",
        "t2i_van_gogh_logits", "t2i_vector_art_logits", "t2i_pixel_art_logits",
        "t2i_watercolor_logits", "t2i_animation_logits",
    ] + list(_T2I_STYLE_TEMPLATES.keys())
    assert mode in _valid_modes, f"Invalid mode: {mode}"

    try:
        data = pickle.loads(data)
        image_bytes_list = data["images"]
        prompts = data["prompts"]

        if mode == "edit_logits":
            # Image editing scoring: ref image + edited image + instruction
            ref_image_bytes_list = data["ref_images"]
            task_types = data.get("task_types", [""] * len(prompts))
            scores = evaluate_edit_images(image_bytes_list, prompts, ref_image_bytes_list, task_types)
        elif mode == "edit_success_logits":
            # Edit success via logit-based scoring on 0-25 scale
            ref_image_bytes_list = data["ref_images"]
            task_types = data.get("task_types", [""] * len(prompts))
            scores = evaluate_edit_success_images(image_bytes_list, prompts, ref_image_bytes_list, task_types)
        elif mode == "edit_preservation_logits":
            # Edit preservation via logit-based scoring on 0-25 scale
            ref_image_bytes_list = data["ref_images"]
            task_types = data.get("task_types", [""] * len(prompts))
            scores = evaluate_edit_preservation_images(image_bytes_list, prompts, ref_image_bytes_list, task_types)
        elif mode == "edit_unified":
            # Unified edit evaluation: returns both success and preservation in one call
            ref_image_bytes_list = data["ref_images"]
            success_scores, preservation_scores = evaluate_edit_unified_images(
                image_bytes_list, prompts, ref_image_bytes_list
            )
            response = {"edit_success_scores": success_scores, "edit_preservation_scores": preservation_scores}
            response = pickle.dumps(response)
            return response, 200
        elif mode == "t2i_logits":
            # Text-to-image scoring: only generated image + prompt, no ref image
            scores = evaluate_t2i_images(image_bytes_list, prompts)
        elif mode == "t2i_style_logits":
            # Text-to-image style scoring: sketch/style adherence
            scores = evaluate_t2i_style_images(image_bytes_list, prompts)
        elif mode == "t2i_photorealism_logits":
            # Text-to-image photorealism scoring
            scores = evaluate_t2i_photorealism_images(image_bytes_list, prompts)
        elif mode == "t2i_van_gogh_logits":
            # Text-to-image Van Gogh style scoring
            scores = evaluate_t2i_van_gogh_images(image_bytes_list, prompts)
        elif mode == "t2i_vector_art_logits":
            # Text-to-image vector art style scoring
            scores = evaluate_t2i_vector_art_images(image_bytes_list, prompts)
        elif mode == "t2i_pixel_art_logits":
            # Text-to-image pixel art style scoring
            scores = evaluate_t2i_pixel_art_images(image_bytes_list, prompts)
        elif mode == "t2i_watercolor_logits":
            # Text-to-image watercolor style scoring
            scores = evaluate_t2i_watercolor_images(image_bytes_list, prompts)
        elif mode == "t2i_animation_logits":
            # Text-to-image animation/cartoon style scoring
            scores = evaluate_t2i_animation_images(image_bytes_list, prompts)
        elif mode in _T2I_STYLE_TEMPLATES:
            # Generic template-driven T2I style scoring
            template_attr = _T2I_STYLE_TEMPLATES[mode]
            scores = evaluate_t2i_generic_style_images(image_bytes_list, prompts, template_attr)
        else:
            # Image editing scoring: ref image + edited image + requirements
            ref_image_bytes_list = data.get("ref_images", None)
            metadatas = data.get("metadatas", [])
            requirements = []
            default_requirement = "preserve the details of the original image exactly and follow the instructions strictly"
            for metadata in metadatas:
                requirements.append(metadata.get("requirement", default_requirement))

            scores = evaluate_images(
                image_bytes_list, prompts, ref_image_bytes_list, requirements
            )

        response = {"scores": scores}
        response = pickle.dumps(response)
        returncode = 200
    except KeyError as e:
        response = f"KeyError: {str(e)}"
        response = response.encode("utf-8")
        returncode = 500
    except Exception:
        response = traceback.format_exc()
        response = response.encode("utf-8")
        returncode = 500

    return response, returncode


if __name__ == "__main__":
    initialize_ray_workers(NUM_GPUS, NUM_TP)
    print(f"Starting Flask server with {NUM_GPUS//NUM_TP} Ray workers...")
    app.run(host="0.0.0.0", port=12341, debug=False, threaded=True)