"""Generate local synthetic format examples, not a research training dataset."""
import argparse
import copy
import json
from pathlib import Path

from PIL import Image, ImageDraw


def make_demo(output, records=2048):
    output = Path(output)
    if records <= 0:
        raise ValueError("records must be positive")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing data directory")
    templates = Path(__file__).resolve().parents[1]/"examples"/"data"
    output.mkdir(parents=True)
    (output/"images").mkdir()
    image = Image.new("RGB", (224,224), "white")
    ImageDraw.Draw(image).rectangle((32,32,191,191), fill="red")
    image.save(output/"images"/"example.png")
    for name in ("tokenizer.jsonl", "pt.jsonl", "text_pt.jsonl", "pt_conversations.jsonl", "sft.json", "text_sft.json"):
        source = templates/name
        if source.suffix == ".jsonl":
            rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        else:
            rows = json.loads(source.read_text())
        samples = []
        for index in range(records):
            row = copy.deepcopy(rows[index % len(rows)])
            prefix = f"Example {index}. "
            if "messages" in row:
                row["messages"][0]["content"] = prefix+row["messages"][0]["content"]
            elif "input" in row:
                row["input"] = prefix+row["input"]
            elif "content" in row:
                row["content"] = prefix+row["content"]
            samples.append(row)
        with (output/source.name).open("w", encoding="utf-8") as handle:
            if source.suffix == ".jsonl":
                for row in samples:
                    handle.write(json.dumps(row)+"\n")
            else:
                json.dump(samples, handle, indent=2)
    print(f"Wrote {records} synthetic examples per source to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/demo")
    parser.add_argument("--records", type=int, default=2048)
    args = parser.parse_args()
    make_demo(args.output, args.records)
