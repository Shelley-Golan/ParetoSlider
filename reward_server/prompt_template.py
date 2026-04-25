SCORE_LOGIT = """Here are two images: the original and the edited version. Please evaluate the edited image based on the requirement.
Instruction: {prompt}
Requirements: "{requirement}"
You need to rate the editing result from 0 to 5 based on requirement: {requirement}.
0: The edit completely fails to meet the requirements.
5: The requirements were met, and the visual result is high quality.
Response Format (Directly response the score number):
0-5"""

SCORE_EDIT_LOGIT = """Here are two images: the original (first) and the edited version (second).
Edit instruction: "{prompt}"
Editing task: "{editing_task}"

Evaluate the editing result on:
1. Instruction following: Does the edited image correctly apply the requested change described in the instruction and follows the editing task?
2. Preservation: Are unrelated details (identity, background, clothing, pose, etc.) preserved from the original except the requested change?
3. Quality: Does the result look natural and high quality (no artifacts, distortions, or unrealistic elements)?

Rate from 0 to 5.
0: Edit completely fails or produces severe artifacts.
1: Edit mostly fails — instruction barely followed or major details lost.
2: Partial success — some aspects of the instruction applied but significant issues remain.
3: Moderate success — instruction mostly followed but some unrelated details changed or minor artifacts present.
4: Good result — instruction followed well and most original details preserved, minor imperfections.
5: Excellent — instruction perfectly followed, unrelated details fully preserved, natural-looking result.
Response Format (Directly response the score number):
0-5"""

SCORE_EDIT_SUCCESS_LOGIT = """Here are two images: the original (first) and the edited version (second).
Edit instruction: "{prompt}"
Editing task type: "{editing_task}"

Evaluate ONLY the editing instruction following: Does the edited image correctly apply the requested change?

Rate from 0 to 9.
0: Edit completely fails to follow the instruction — no visible change or wrong change applied.
1: Minimal attempt — very slight or barely noticeable change in the right direction.
2: Edit mostly fails — instruction barely followed with major issues.
3: Weak attempt — some intent visible but significant errors or incompleteness.
4: Partial success — some aspects of the instruction applied but significant issues remain.
5: Moderate success — instruction mostly followed but with clear imperfections.
6: Good attempt — instruction well followed with only minor issues.
7: Good result — instruction accurately followed with very minor imperfections.
8: Very good — instruction followed very well, nearly perfect execution.
9: Excellent — instruction perfectly followed with high quality result.
Response Format (Directly response the score number):
0-9"""

SCORE_T2I_LOGIT = """Here is a generated image based on the following text caption.
Text Caption: "{prompt}"
Evaluate the image on:
Alignment: Does the image accurately depict the objects, attributes, and relationships described in the caption?
Rate from 0 to 5.
0: Completely fails to match the caption or is very low quality.
5: Perfectly matches the caption with excellent visual quality.
Response Format (Directly response the score number):
0-5"""


SCORE_T2I_LOGIT_SKETCH = """You are judging a generated image against a caption.
Caption: "A sketch of {prompt}. White background. Line-art style."

Task:
Give ONE integer score from 0 to 5 based on BOTH content alignment and demanded sketch style.
Be strict. Do not guess unseen details.

Step 1) Style checks (each is pass/fail):
- Line-based: depiction is primarily lines (no solid filled regions).
- Pencil-like: looks like pencil/graphite strokes (line-weight variation is ok).
- No background.
- Black and white strokes.
- No realistic features at all. Only pencil/graphite strokes are allowed.
- No realistic textures or materials.

Step 2) Content checks (each is pass/fail):
- Main subject(s) in {prompt} clearly present and recognizable.
- Key attributes from {prompt} present.
- Key relationships/actions from {prompt} are correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; image is clear.
- 4: Style passes AND content mostly correct (minor missing attribute or slightly off relation).
- 3: Style passes but content has noticeable errors OR content correct but one style check fails.
- 2: Partial match: multiple content errors and/or multiple style failures, but some intent is visible.
- 1: Very weak match: only tiny fragments match; style and content largely wrong.
- 0: Completely wrong style and content, or image is unusable (blank, corrupted, unreadable).

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""


SCORE_T2I_LOGIT_PHOTOREALISM = """You are judging a generated image against a caption.
Caption: "A photorealistic photo of {prompt}."

Task:
Give ONE integer score from 0 to 5 based on BOTH content alignment and photorealistic style.
Be strict. Do not hallucinate details.

Step 1) Style checks (pass/fail):
- Looks like a real photo (not illustration, not cartoon, not 3D render).
- Realistic lighting and shadows (consistent with a single scene).
- Realistic textures/materials (skin, fabric, metal, wood, etc. look natural if present).
- No painterly brush strokes, no heavy outlines, no flat shading.

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} clearly present.
- Key attributes from {prompt} present (count, colors, distinctive parts).
- Key relationships/actions from {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; image is clear and detailed.
- 4: Style passes AND content mostly correct with only minor issues.
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but one style check fails.
- 2: Multiple content mistakes and/or multiple style failures, but some intent visible.
- 1: Very weak match; most requirements unmet.
- 0: Totally wrong or unusable image.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""

SCORE_T2I_LOGIT_VECTOR_ART = """You are judging a generated image against a caption.
Caption: "Geometric abstraction of {prompt} composed of smooth geometric shapes, flat colors, clean edges, squares, circles, triangles, etc."

Task:
Give ONE integer score from 0 to 5 based on BOTH content alignment and demanded vector style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Flat colors (no realistic textures; no photographic detail).
- Clean hard edges (not painterly; not blurry).
- Geometric, simplified shapes (icon-like abstraction is ok, focus on squares, circles, triangles, etc.).
- Minimalistic composition (limited detail; not overly complex).
- No gradients (if gradients are present, treat as a style failure unless very subtle).

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} clearly present.
- Key attributes from {prompt} present where compatible (counts, main parts).
- Key relationships/actions from {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; crisp and readable.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable image.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""


SCORE_T2I_LOGIT_PIXEL_ART = """You are judging a generated image against a caption.
Caption: "Pixel art of {prompt}. Low-resolution retro style. Visible square pixels. Limited color palette."

Task:
Output ONE integer score 0 to 5 for BOTH content alignment and demanded pixel-art style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Visible square pixels (blocky edges, not smooth).
- Low-resolution look (sprite-like), not vector clean, not pencil sketch.
- Limited palette (few colors), no realistic textures.

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} present and recognizable.
- Key attributes and relationships in {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; crisp and readable.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable image.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""


SCORE_T2I_LOGIT_WATERCOLOR = """You are judging a generated image against a caption.
Caption: "Watercolor painting of {prompt}. Soft, painterly brush strokes."

Task:
Output ONE integer score 0 to 5 for BOTH content alignment and demanded watercolor style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Soft, painterly brush strokes.

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} present and recognizable.
- Key attributes and relationships in {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; crisp and readable.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable image.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""


SCORE_T2I_LOGIT_ANIMATION = """You are judging a generated image against a caption.
Caption: "An animated cartoon-style image of {prompt}. Vibrant flat colors, clean outlines, stylized characters."

Task:
Give ONE integer score from 0 to 5 based on BOTH content alignment and demanded animation/cartoon style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Animated / cartoon look (NOT photorealistic, NOT pencil sketch, NOT pixel art).
- Vibrant, saturated flat colors (minimal realistic textures or gradients).
- Clean outlines or stylized shapes (cel-shaded or vector-like).
- Characters/objects are stylized (exaggerated proportions, simplified features).

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} clearly present and recognizable.
- Key attributes from {prompt} present.
- Key relationships/actions from {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; image is clear.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable image.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""


SCORE_T2I_LOGIT_OIL_PAINTING = """You are judging a generated image against a caption.
Caption: "An oil painting of {prompt}. Rich, textured brush strokes with visible impasto. Deep, saturated colors."

Task:
Give ONE integer score from 0 to 5 based on BOTH content alignment and demanded oil painting style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Visible brush strokes (thick, textured, impasto-like).
- Rich, deep color palette (saturated pigments, not flat digital colors).
- Painterly look (NOT photorealistic, NOT vector, NOT pencil sketch).
- Canvas-like texture or blended pigment appearance.

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} clearly present and recognizable.
- Key attributes from {prompt} present.
- Key relationships/actions from {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; image is clear.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable image.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""


SCORE_T2I_LOGIT_ANIME = """You are judging a generated image against a caption.
Caption: "An anime illustration of {prompt}. Japanese animation style with large expressive eyes, clean lines, and vibrant colors."

Task:
Give ONE integer score from 0 to 5 based on BOTH content alignment and demanded anime style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Anime/manga aesthetic (large eyes, stylized facial features, pointed chins).
- Clean line art with smooth outlines (not sketchy, not photorealistic).
- Vibrant, saturated colors with cel-shading or flat coloring.
- Japanese animation look (NOT Western cartoon, NOT realistic, NOT pixel art).

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} clearly present and recognizable.
- Key attributes from {prompt} present.
- Key relationships/actions from {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; image is clear.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable image.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""


SCORE_T2I_LOGIT_FLAT_VECTOR = """You are judging a generated image against a caption.
Caption: "A flat vector illustration of {prompt}. Clean shapes, solid fill colors, no gradients, minimal detail."

Task:
Give ONE integer score from 0 to 5 based on BOTH content alignment and demanded flat vector style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Flat, solid fill colors (no realistic textures, no photographic detail).
- Clean hard edges (not painterly, not blurry, not sketchy).
- Simplified, icon-like shapes (reduced detail, graphic design aesthetic).
- No gradients or at most very subtle ones; no 3D shading.

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} clearly present and recognizable.
- Key attributes from {prompt} present where compatible.
- Key relationships/actions from {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; crisp and readable.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable image.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""


SCORE_T2I_LOGIT_VINTAGE_FILM = """You are judging a generated image against a caption.
Caption: "A vintage film photograph of {prompt}. Faded colors, film grain, light leaks, retro analog camera look."

Task:
Give ONE integer score from 0 to 5 based on BOTH content alignment and demanded vintage film style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Faded, desaturated, or warm-shifted color palette (not vivid digital colors).
- Visible film grain or analog noise texture.
- Vintage analog look (light leaks, vignetting, or soft focus are acceptable markers).
- NOT a modern digital photo look; NOT illustration or painting.

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} clearly present and recognizable.
- Key attributes from {prompt} present.
- Key relationships/actions from {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; image is clear.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable image.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""


SCORE_T2I_LOGIT_NEON_CYBERPUNK = """You are judging a generated image against a caption.
Caption: "A cyberpunk scene of {prompt}. Bright neon glowing colors, dark atmosphere, futuristic urban aesthetic."

Task:
Give ONE integer score from 0 to 5 based on BOTH content alignment and demanded neon cyberpunk style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Bright neon colors (pink, cyan, purple, electric blue) dominating the palette.
- Dark or moody atmosphere (nighttime, shadows, high contrast).
- Futuristic / sci-fi aesthetic (neon signs, holographic elements, tech-inspired).
- NOT natural daylight photo; NOT pastel; NOT traditional painting.

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} clearly present and recognizable.
- Key attributes from {prompt} present.
- Key relationships/actions from {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; image is clear.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable image.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""


SCORE_T2I_LOGIT_LOW_POLY = """You are judging a generated image against a caption.
Caption: "Low-poly 3D art of {prompt}. Flat-shaded triangular facets, geometric surfaces, minimal detail."

Task:
Give ONE integer score from 0 to 5 based on BOTH content alignment and demanded low-poly style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Visible polygonal facets (triangular or geometric flat surfaces).
- Flat shading per face (no smooth gradients across surfaces).
- Simplified geometric forms (reduced vertex count, angular silhouettes).
- NOT photorealistic; NOT smooth 3D render; NOT 2D illustration.

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} clearly present and recognizable.
- Key attributes from {prompt} present.
- Key relationships/actions from {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; crisp and readable.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable image.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""


SCORE_T2I_LOGIT_UKIYOE = """You are judging a generated image against a caption.
Caption: "Ukiyo-e Japanese woodblock print of {prompt}. Bold outlines, flat color areas, traditional Japanese composition."

Task:
Give ONE integer score from 0 to 5 based on BOTH content alignment and demanded ukiyo-e style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Bold black outlines defining forms (woodblock print line quality).
- Flat color areas with minimal shading (no photorealistic gradients).
- Traditional Japanese aesthetic (composition, subject treatment, color palette).
- Looks like a woodblock print reproduction (NOT anime, NOT watercolor, NOT photo).

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} clearly present and recognizable.
- Key attributes from {prompt} present.
- Key relationships/actions from {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; image is clear.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable image.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""


SCORE_T2I_LOGIT_PASTEL = """You are judging a generated image against a caption.
Caption: "Soft pastel artwork of {prompt}. Muted gentle colors, chalky texture, delicate blending."

Task:
Give ONE integer score from 0 to 5 based on BOTH content alignment and demanded pastel style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Muted, soft, desaturated color palette (pastel tones, not vivid or neon).
- Chalky or powdery texture (visible grain of pastel medium, soft edges).
- Gentle blending with soft transitions (not hard edges, not flat digital colors).
- NOT photorealistic; NOT vector; NOT oil painting with thick impasto.

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} clearly present and recognizable.
- Key attributes from {prompt} present.
- Key relationships/actions from {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; image is clear.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable image.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""


SCORE_T2I_LOGIT_DISNEY = """You are judging a generated image against a caption.
Caption: "A Disney animated scene of {prompt}. Polished 3D or classic Disney hand-drawn style, expressive characters, warm lighting."

Task:
Give ONE integer score from 0 to 5 based on BOTH content alignment and demanded Disney animation style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Disney animation aesthetic (polished, appealing character design with large expressive eyes).
- Warm, inviting color palette and lighting (golden tones, soft shadows).
- Either classic hand-drawn Disney look OR modern Disney/Pixar 3D render style.
- NOT anime; NOT dark/gothic; NOT photorealistic; NOT pixel art.

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} clearly present and recognizable.
- Key attributes from {prompt} present.
- Key relationships/actions from {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; image is clear.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable image.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""


SCORE_T2I_LOGIT_COMIC_BOOK = """You are judging a generated image against a caption.
Caption: "A comic book illustration of {prompt}. Bold ink outlines, halftone dots, dynamic composition."

Task:
Give ONE integer score from 0 to 5 based on BOTH content alignment and demanded comic book style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Bold black ink outlines (strong contour lines defining shapes).
- Comic book coloring (flat or halftone-dotted fills, limited shading).
- Dynamic, action-oriented composition (dramatic angles or poses).
- Looks like a comic book panel (NOT anime, NOT photorealistic, NOT watercolor).

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} clearly present and recognizable.
- Key attributes from {prompt} present.
- Key relationships/actions from {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; image is clear.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable image.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""


SCORE_T2I_LOGIT_CARICATURE = """You are judging a generated image against a caption.
Caption: "A caricature of {prompt}. Exaggerated facial features and body proportions, humorous stylization."

Task:
Give ONE integer score from 0 to 5 based on BOTH content alignment and demanded caricature style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Exaggerated proportions (oversized head, enlarged or distorted features).
- Humorous or satirical stylization (not realistic proportions).
- Hand-drawn or illustrated look (not photorealistic, not 3D render).
- Recognizable as a caricature (NOT anime, NOT standard cartoon, NOT realistic portrait).

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} clearly present and recognizable.
- Key attributes from {prompt} present.
- Key relationships/actions from {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; image is clear.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable image.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""


SCORE_T2I_LOGIT_ORIGAMI = """You are judging a generated image against a caption.
Caption: "Origami paper craft of {prompt}. Folded paper with visible creases, geometric faceted shapes, clean paper texture."

Task:
Give ONE integer score from 0 to 5 based on BOTH content alignment and demanded origami style.
Be strict. Do not guess unseen details.

Step 1) Style checks (pass/fail):
- Looks like folded paper (visible creases, angular folds, paper edges).
- Geometric, faceted surfaces (paper-like flat planes, not smooth curves).
- Clean paper texture (matte, solid-colored paper material appearance).
- NOT a drawing of origami; must look like actual 3D paper craft or a convincing render of one.

Step 2) Content checks (pass/fail):
- Main subject(s) in {prompt} clearly present and recognizable as origami form.
- Key attributes from {prompt} present where compatible with paper folding.
- Key relationships/actions from {prompt} correct (if any).

Scoring rule:
- 5: All style checks pass AND all content checks pass; crisp and readable.
- 4: Style passes AND content mostly correct (minor missing attribute/detail).
- 3: Either (A) style passes but content has clear mistakes, or (B) content correct but 1 style check fails.
- 2: Partial match; multiple failures but some intent visible.
- 1: Very weak match.
- 0: Totally wrong or unusable image.

Response format: output ONLY the single digit 0 1 2 3 4 or 5."""
