"""Build visual sheets to manually inspect the 25 GT boxes; not used for OCR."""

import csv
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    with Path("metadata/plates.csv").open(newline="", encoding="utf-8-sig") as fp:
        rows=[r for r in csv.DictReader(fp) if r["split"]=="test"]
    grouped=defaultdict(list)
    for r in rows:
        grouped[r["image"]].append(r)
    output=Path("results/label_review")
    output.mkdir(parents=True, exist_ok=True)
    images=sorted(grouped)
    for start in range(0,len(images),5):
        sheet=Image.new("RGB",(1350,5*250),"#202020")
        draw=ImageDraw.Draw(sheet)
        for index, path in enumerate(images[start:start+5]):
            img=Image.open(path).convert("RGB")
            marked=img.copy()
            ink=ImageDraw.Draw(marked)
            boxes=[]
            for row in grouped[path]:
                box=[int(row[k]) for k in ("bbox_x1","bbox_y1","bbox_x2","bbox_y2")]
                ink.rectangle(box,outline="red",width=max(3,img.width//900))
                boxes.append(box)
            marked.thumbnail((600,205))
            y=index*250
            sheet.paste(marked,(0,y+35))
            draw.text((10,y+8),f"{Path(path).name} | {len(boxes)} boxes",fill="white")
            for i,box in enumerate(boxes[:3]):
                crop=img.crop(box)
                crop.thumbnail((225,130))
                xx=615+i*240
                sheet.paste(crop,(xx,y+48))
                draw.text((xx,y+9),f"plate {i}: {box}",fill="white")
        sheet.save(output/f"test_labels_page_{start//5+1:02d}.jpg",quality=88)
    print(f"Wrote {(len(images)+4)//5} label review sheets to {output}")


if __name__ == "__main__":
    main()
