import os
import torch
import numpy as np
from typing import List
from vllm import LLM, SamplingParams
import vllm
from PIL import Image
from io import BytesIO
import base64
import pickle
import traceback
from flask import Flask, request
import ray
import asyncio
import prompt_template
import lpips
from skimage.metrics import structural_similarity as ssim

if vllm.__version__ != "0.9.2":
    raise ValueError("vLLM version must be 0.9.2")

os.environ["VLLM_USE_V1"] = "0"  # IMPORTANT

app = Flask(__name__)

# Global variables
score_idx = [15, 16, 17, 18, 19, 20]
workers = []  # Ray actors for each GPU
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
NUM_GPUS = 1
NUM_TP = 1
lpips_model = None  # Global LPIPS model (loaded once)

def get_base64(image):
    image_data = BytesIO()
    image.save(image_data, format="JPEG")
    image_data_bytes = image_data.getvalue()
    encoded_image = base64.b64encode(image_data_bytes).decode("utf-8")
    return encoded_image


def init_lpips():
    """Initialize the LPIPS model (call once at startup)"""
    global lpips_model
    lpips_model = lpips.LPIPS(net='alex')  # AlexNet is faster, VGG is slightly better
    lpips_model.eval()
    print("LPIPS model initialized")


def compute_lpips_similarity(image_bytes, ref_image_bytes):
    """Compute LPIPS-based perceptual similarity (0-1 range, higher = more similar)"""
    global lpips_model
    
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    ref_image = Image.open(BytesIO(ref_image_bytes)).convert("RGB")
    
    # Resize to 224x224 for LPIPS
    image = image.resize((224, 224))
    ref_image = ref_image.resize((224, 224))
    
    # Convert to tensor and normalize to [-1, 1]
    img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1
    ref_tensor = torch.tensor(np.array(ref_image)).permute(2, 0, 1).float() / 127.5 - 1
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    ref_tensor = ref_tensor.unsqueeze(0)
    
    with torch.no_grad():
        distance = lpips_model(img_tensor, ref_tensor).item()
    
    # Convert distance to similarity (LPIPS distance is typically 0-1)
    similarity = max(0.0, 1.0 - distance)
    return similarity


def compute_ssim_similarity(image_bytes, ref_image_bytes):
    """Compute SSIM-based structural similarity (0-1 range, higher = more similar)"""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    ref_image = Image.open(BytesIO(ref_image_bytes)).convert("RGB")
    
    # Resize to same dimensions if needed
    if image.size != ref_image.size:
        image = image.resize(ref_image.size)
    
    img_arr = np.array(image)
    ref_arr = np.array(ref_image)
    
    # Compute SSIM (returns value in [-1, 1], typically [0, 1] for real images)
    # channel_axis=2 for RGB images
    score = ssim(img_arr, ref_arr, channel_axis=2, data_range=255)
    
    # Ensure in [0, 1] range
    return max(0.0, score)


class LogitsSpy:
    def __init__(self):
        self.processed_logits: list[torch.Tensor] = []

    def __call__(self, token_ids: list[int], logits: torch.Tensor):
        self.processed_logits.append(logits)
        return logits


@ray.remote(num_gpus=NUM_TP)
class ModelWorker:
    def __init__(self):
        self.llm = None
        self.load_model()

    def load_model(self):
        """Load the Qwen2-VL model using vLLM on specific GPU"""
        self.llm = LLM(
            MODEL_PATH, limit_mm_per_prompt={"image": 3}, tensor_parallel_size=NUM_TP
        )

    def evaluate_image(
        self, image_bytes, prompt, ref_image_bytes=None, requirement: str = ""
    ):
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_bytes), formats=["jpeg"])
        ref_image = Image.open(BytesIO(ref_image_bytes), formats=["jpeg"])
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": ref_image},
                    {"type": "image_pil", "image_pil": image},
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

    def _vllm_evaluate(self, conversation, max_tokens=3, max_score=5):
        logits_spy = LogitsSpy()
        sampling_params = SamplingParams(
            max_tokens=max_tokens, logits_processors=[logits_spy]
        )
        self.llm.chat(conversation, sampling_params=sampling_params)
        if not logits_spy.processed_logits:
            raise RuntimeError("_vllm_evaluate: no logits received from model")
        probs = torch.softmax(logits_spy.processed_logits[0][score_idx], dim=-1)
        score_prob = (
            torch.sum(
                probs * torch.arange(len(score_idx)).to(probs.device)
            ).item()
            / max_score
        )
        print(f"Score: {score_prob:.4f}")
        return score_prob


def initialize_ray_workers(num_gpus=8, num_tp=4):
    global workers
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

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


@app.route("/mode/<mode>", methods=["POST"])
def inference_mode(mode):
    data = request.get_data()

    assert mode in ["logits_non_cot"], "Invalid mode"

    try:
        data = pickle.loads(data)
        image_bytes_list = data["images"]
        ref_image_bytes_list = data.get("ref_images", None)
        prompts = data["prompts"]

        # Requirement 1: Preservation (MLLM + LPIPS + SSIM)
        preserve_requirements = ["Ignore the instruction! Score only if the output is exactly the same as the reference image!"] * len(prompts)
        preserve_scores = evaluate_images(
            image_bytes_list, prompts, ref_image_bytes_list, preserve_requirements
        )
        
        # Compute perceptual similarity scores (LPIPS + SSIM)
        lpips_scores = [
            compute_lpips_similarity(img_bytes, ref_bytes)
            for img_bytes, ref_bytes in zip(image_bytes_list, ref_image_bytes_list)
        ]
        ssim_scores = [
            compute_ssim_similarity(img_bytes, ref_bytes)
            for img_bytes, ref_bytes in zip(image_bytes_list, ref_image_bytes_list)
        ]
        
        # Combine: average of LPIPS and SSIM (both in 0-1 range, higher = more similar)
        perceptual_scores = [(lpips + ssim_val) / 2 for lpips, ssim_val in zip(lpips_scores, ssim_scores)]
        preserve_scores = [mllm + perceptual for mllm, perceptual in zip(preserve_scores, perceptual_scores)]
        
        # Requirement 2: Instruction following
        instruct_requirements = ["follow the instructions strictly"] * len(prompts)
        instruct_scores = evaluate_images(
            image_bytes_list, prompts, ref_image_bytes_list, instruct_requirements
        )

        response = {
            "scores": [p + i for p, i in zip(preserve_scores, instruct_scores)],  # element-wise sum
            "preserve_score": perceptual_scores,
            "lpips_scores": lpips_scores,
            "ssim_scores": ssim_scores,
            "instruct_follow_score": instruct_scores,
        }
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
    init_lpips()
    initialize_ray_workers(NUM_GPUS, NUM_TP)
    print(f"Starting Flask server with {NUM_GPUS//NUM_TP} Ray workers...")
    app.run(host="0.0.0.0", port=12341, debug=False)