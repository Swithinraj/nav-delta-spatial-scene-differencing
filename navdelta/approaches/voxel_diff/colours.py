"""Step 4c (optional): colour adjectives from a small open VLM.

SmolVLM2-500M-Video-Instruct (Apache-2.0, ~1.1 GB in fp16, fits the 4 GB GPU)
is asked one narrow question per view of each labelled object: "what is its
main colour?". Only the colour is used. In a test on this scene the model's
colour words were right on every object, while its shape words were not
("spherical" for a cylindrical lamp), so nothing else it says is kept.

Crops are rebuilt from the Step 3 view records (same box, rotated upright)
without the coloured mask outline, so the outline cannot bias the answer.
Answers outside a basic colour vocabulary are discarded; the colour needs a
strict majority of the views (both ends of a move pooled), otherwise no
adjective is used.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from navdelta.approaches.voxel_diff.paths import DATA, OUTPUT, SEMANTICS, inside_workspace

import numpy as np
import torch
from PIL import Image

from navdelta.approaches.voxel_diff.labels import TIGHT_PAD, expand  # noqa: E402
from navdelta.common.device import DEVICE, DTYPE, PRECISION  # noqa: E402
VLM_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
VIEWS_PER_OBJECT = 3
COLOURS = ["black", "white", "grey", "gray", "silver", "brown", "beige", "cream", "red", "orange", "yellow",
           "gold", "green", "blue", "navy", "dark blue", "light blue", "purple", "pink", "transparent", "clear",
           "dark grey", "light grey", "dark brown", "light brown"]


def normalise_colour(text: str) -> str | None:
    words = "".join(c if c.isalpha() or c == " " else " " for c in text.lower()).split()
    for size in (2, 1):
        for i in range(len(words) - size + 1):
            phrase = " ".join(words[i:i + size])
            if phrase in COLOURS:
                return {"gray": "grey", "clear": "transparent", "navy": "dark blue"}.get(phrase, phrase)
    return None


def main() -> None:
    started = time.time()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    labels = json.loads(inside_workspace(SEMANTICS / "object_labels.json").read_text())
    processor = AutoProcessor.from_pretrained(VLM_MODEL, revision=VLM_REVISION)
    model = AutoModelForImageTextToText.from_pretrained(VLM_MODEL, revision=VLM_REVISION, dtype=DTYPE).to(DEVICE).eval()

    answers: dict[str, list[dict]] = {}
    for record in labels["objects"]:
        answers[record["object"]] = []
        for view in record["views"][:VIEWS_PER_OBJECT]:
            image = np.asarray(Image.open(inside_workspace(DATA / view["rgb"])).convert("RGB"))
            x0, y0, x1, y1 = expand(view["box_xyxy"], TIGHT_PAD, image.shape[1], image.shape[0])
            crop = np.ascontiguousarray(np.rot90(image[y0:y1, x0:x1], view["upright_turns_ccw"]))
            question = (f"This photo shows a {record['label']}. What is its main colour? "
                        f"Answer with one or two words.")
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[Image.fromarray(crop)], return_tensors="pt").to(
                DEVICE, dtype=DTYPE)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            text = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            answers[record["object"]].append({"frame": view["frame"], "answer": text.strip(),
                                              "colour": normalise_colour(text)})
    del model
    torch.cuda.empty_cache()

    def decide(views: list[dict]) -> dict:
        votes = Counter(v["colour"] for v in views if v["colour"])
        colour = None
        if votes:
            best, count = votes.most_common(1)[0]
            if count > len(views) / 2:
                colour = best
        return {"colour": colour, "votes": dict(votes), "views": len(views)}

    per_object = {key: {**decide(views), "answers": views} for key, views in answers.items()}
    per_event = {}
    for event in labels["events"]:
        members = [r["object"] for r in labels["objects"] if r["event"] == event["event"]]
        per_event[event["event"]] = decide([v for m in members for v in answers[m]])

    output = {
        "schema_version": "1.0.0",
        "stage": "step4c_optional_vlm_colour",
        "learned_models_used": [{"id": VLM_MODEL, "revision": VLM_REVISION, "license": "Apache-2.0",
                                 "use": "one question per view: main colour", "trained_or_finetuned_here": False}],
        "generation": {"do_sample": False, "max_new_tokens": 8, "precision": PRECISION},
        "colour_vocabulary": COLOURS,
        "events": per_event,
        "objects": per_object,
    }
    output["result_sha256"] = hashlib.sha256(json.dumps({"events": per_event, "objects": per_object},
                                                        sort_keys=True).encode()).hexdigest()
    output["runtime_s"] = round(time.time() - started, 1)
    (inside_workspace(SEMANTICS) / "appearance.json").write_text(json.dumps(output, indent=2))
    for event, result in per_event.items():
        print(f"  {event:<5} {result['colour']!s:<12} votes {result['votes']} of {result['views']} views")
    print(f"Result sha256: {output['result_sha256']}")
    print(f"Done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
