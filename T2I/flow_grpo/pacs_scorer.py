"""
PACS Domain Scorer for photorealism vs drawnness classification.

Uses the PACS-DG-SigLIP2 model to classify images into:
- photo (photorealistic)
- art_painting
- cartoon  
- sketch

These are used to compute photorealism and drawnness scores for Pareto optimization.
"""

import torch
class PACSScorer:
    """Scorer for photorealism vs drawnness using PACS domain classification."""
    
    def __init__(self, device="cuda"):
        self.device = device
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        self.model = AutoModelForImageClassification.from_pretrained("prithivMLmods/PACS-DG-SigLIP2").to(device)
        self.processor = AutoImageProcessor.from_pretrained("prithivMLmods/PACS-DG-SigLIP2")
        self.label2id = self.model.config.label2id
    
    def score(self, images):
        """
        Compute photorealism and drawnness scores for each image.
        
        Returns scaled raw logits (x0.1) instead of softmax probabilities to prevent
        reward saturation. Raw logits provide smoother gradients for multi-objective
        optimization (e.g., with PickScore).
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Scale logits by 0.1 and apply softplus to ensure positive scores
        # softplus(x) = log(1 + exp(x)) - smooth, always positive, no gradient clipping
        # Raw logits typically range from -10 to +20, output becomes ~(0, 3)
        # This prevents reward saturation that occurs with softmax probabilities
        scaled_logits = torch.nn.functional.softplus(logits) / 10.0
        
        results = []
        photo_idx = self.label2id.get("photo")
        art_idx = self.label2id.get("art_painting")
        cartoon_idx = self.label2id.get("cartoon")
        sketch_idx = self.label2id.get("sketch")
        
        for i in range(len(images)):
            photorealism_score = scaled_logits[i, photo_idx].item()
            # Use only art_painting logit for drawnness (more specific style)
            art_painting_score = scaled_logits[i, art_idx].item()
            cartoon_score = scaled_logits[i, cartoon_idx].item()
            sketch_score = scaled_logits[i, sketch_idx].item()

            total_art_score = max(art_painting_score, cartoon_score, sketch_score)
            
            results.append((photorealism_score, total_art_score))
        
        return results

    @torch.no_grad()
    def score_sketch_evidence(self, images):
        """Returns sketch evidence (softplus-scaled logit / 10) for each PIL image.
        
        Uses the same scaling convention as score() for consistency.
        Returns a tensor [N] on device instead of a list of scalars for efficiency.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        scaled = torch.nn.functional.softplus(logits) / 10.0
        sketch_idx = self.label2id.get("sketch")
        if sketch_idx is None:
            raise KeyError(
                f"'sketch' not found in model label2id. Available labels: {list(self.label2id.keys())}"
            )
        return scaled[:, sketch_idx]  # tensor [N] on device

    @torch.no_grad()
    def score_label_evidence(self, images, label):
        """Returns evidence (softplus-scaled logit / 10) for specified label.
        
        Generalization of score_sketch_evidence for any PACS label.
        
        Args:
            images: list of PIL images
            label: one of "photo", "art_painting", "cartoon", "sketch"
        Returns:
            tensor [N] on device
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        scaled = torch.nn.functional.softplus(logits) / 10.0
        label_idx = self.label2id.get(label)
        if label_idx is None:
            raise KeyError(
                f"'{label}' not found in model label2id. Available labels: {list(self.label2id.keys())}"
            )
        return scaled[:, label_idx]

    def score_photo_nonphoto(self, images):
        """
        Compute photorealism and non_photo scores for each image.
        
        Uses scaled logits (same as score()) for consistent scale and smooth gradients:
        - photo = softplus(photo_logit) / 10
        - non_photo = sum of softplus(art, cartoon, sketch logits) / 10
        
        This avoids softmax gradient saturation while combining all non-photo styles.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        photo_idx = self.label2id.get("photo")
        art_idx = self.label2id.get("art_painting")
        cartoon_idx = self.label2id.get("cartoon")
        sketch_idx = self.label2id.get("sketch")
        
        # Use same scaling as score() for consistency
        scaled_logits = torch.nn.functional.softplus(logits) / 10.0
        
        results = []
        for i in range(len(images)):
            photo_score = scaled_logits[i, photo_idx].item()
            # Average of all non-photo styles (divided by 3 for same scale as photo)
            non_photo_score = (
                scaled_logits[i, art_idx].item() +
                scaled_logits[i, cartoon_idx].item() +
                scaled_logits[i, sketch_idx].item()
            ) / 3.0
            results.append((photo_score, non_photo_score))
        
        return results
    
    def score_batch(self, images):
        """
        Batch scoring for efficiency.
        """
        results = self.score(images)
        return {
            "photorealism": [r[0] for r in results],
            "drawnness": [r[1] for r in results],
        }
