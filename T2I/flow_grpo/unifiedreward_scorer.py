import asyncio
from openai import AsyncOpenAI
import base64
from io import BytesIO
import re
from PIL import Image


def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    return base64_qwen


def _extract_scores(text_outputs):
    scores = []
    pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
    for text in text_outputs:
        match = re.search(pattern, text)
        if match:
            try:
                scores.append(float(match.group(1)))
            except ValueError:
                scores.append(0.0)
        else:
            scores.append(0.0)
    return scores


client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")


async def evaluate_image(prompt, image):
    question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
    images_base64 = pil_image_to_base64(image)
    response = await client.chat.completions.create(
        model="UnifiedReward-7b-v1.5",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": images_base64},
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content


async def evaluate_batch_image(images, prompts):
    tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
    results = await asyncio.gather(*tasks)
    return results


async def evaluate_image_style(style_prompt, image):
    question = (
        "<image>\n"
        "You are given a style description and a generated image. "
        "Your task is to evaluate this image ONLY on style.\n\n"
        "Focus on these criteria:\n"
        "1. Style Match: How well the image reflects the intended style described in the text.\n"
        "2. Visual Appeal: The overall aesthetic quality of the image.\n"
        "3. Stylistic Consistency: Whether the style is applied consistently across the image.\n"
        "4. Quality: Clarity, detail preservation, color harmony, and absence of visual artifacts.\n\n"
        "Give a brief explanation, then output exactly one line in this format:\n"
        "Final Score: <number from 1 to 5>\n\n"
        f"Style Description: [{style_prompt}]"
    )

    images_base64 = pil_image_to_base64(image)

    response = await client.chat.completions.create(
        model="UnifiedReward-7b-v1.5",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": images_base64},
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            },
        ],
        temperature=0,
    )

    content = response.choices[0].message.content

    match = re.search(r"Final Score:\s*([1-5](?:\.\d+)?)", content)
    score = float(match.group(1)) if match else None

    return {
        "raw_response": content,
        "style_score": score,
    }


async def evaluate_batch_image_style(images, style_prompts):
    tasks = [
        evaluate_image_style(style_prompt, img)
        for style_prompt, img in zip(style_prompts, images)
    ]
    return await asyncio.gather(*tasks)
    

# Usage example
def main():
    images = [
        "test_cases/nasa.jpg",
        "test_cases/hello world.jpg",
        "test_cases/a photo of a brown giraffe and a white stop sign.png",
    ]
    pil_images = [Image.open(img) for img in images]
    prompts = [
        'An astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
        'New York Skyline with "Hello World" written with fireworks on the sky',
        "a photo of a brown giraffe and a white stop sign",
    ]
    text_outputs = asyncio.run(evaluate_batch_image(pil_images, prompts))
    print(text_outputs)
    score = _extract_scores(text_outputs)
    score = [sc / 5.0 for sc in score]
    print(score)


if __name__ == "__main__":
    main()
