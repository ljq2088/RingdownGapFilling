from pathlib import Path
import fitz
from PIL import Image, ImageOps, ImageDraw
import json

p = Path(__file__).resolve().parent
d = fitz.open(p/'DGF_research_replacement_20261015.pdf')
canvas = Image.new('RGB', (1000, ((len(d)+3)//4)*365), '#dae0e7')
draw = ImageDraw.Draw(canvas)
pages = []
for i, page in enumerate(d):
    image_path = p/'rendered'/f'narrative-{i+1:02}.png'
    page.get_pixmap(matrix=fitz.Matrix(1.3, 1.3)).save(str(image_path))
    canvas.paste(ImageOps.contain(Image.open(image_path), (235, 335)), ((i%4)*250+7, (i//4)*365+20))
    draw.text(((i%4)*250+10, (i//4)*365+3), str(i+1), fill='black')
    pages.append({'page':i+1, 'first_text':page.get_text()[:85].replace('\n',' ')})
canvas.save(p/'rendered/narrative-contact.png')
(p/'narrative_pages.json').write_text(json.dumps(pages, indent=2))
print(json.dumps({'page_count':len(d),'pages':pages}, indent=2))
