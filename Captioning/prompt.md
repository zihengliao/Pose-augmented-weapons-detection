# VLM Caption Prompt for Pose-Augmented Weapons Detection

## Primary Prompt

Generate a structured caption for this weapons detection image. **Keep the total caption under 20 words.**

**Required Format:**
"Scene contains [hand count] hands in [detailed pose description] with [gun presence/absence and details]. [Detection challenges or notable features]."

## Detection Categories to Consider

**Hand Poses/Grips:**
- Open palm, closed fist, pointing, gripping, extended, relaxed
- Single-handed, two-handed, overlapping hands
- Partial visibility, fully visible

**Gun Types/Positions:**
- Pistol, rifle, partially visible weapon
- Held, holstered, on surface, in-hand
- Horizontal, vertical, angled orientation

**Detection Challenges:**
- Occlusion (partial blocking)
- Low lighting, shadows
- Motion blur, unusual angle
- Hand-weapon overlap
- Background clutter

## Example Outputs

Good: "Scene contains two hands gripping a pistol in vertical orientation with partial weapon occlusion by background objects."
Good: "Scene contains one open hand reaching toward holstered weapon with challenging lighting conditions throughout image."
Good: "Scene contains three hands in pointing gestures with rifle visible horizontally, hands overlapping weapon grip area."
Good: "Scene contains multiple hands in various poses with pistol weapon partially occluded by background objects creating detection difficulties."

Too short: "Two hands, gun present, occluded"
Too robotic: "Hands: 2, Gun: pistol, Challenge: occlusion"
