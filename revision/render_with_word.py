"""Native Word fallback after packaged render_docx.py failed (LibreOffice absent).
Creates QA PDF/PNG copies only. Does not save changes to DOCX inputs.
"""
from pathlib import Path
import json
import win32com.client, pythoncom
import pymupdf
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parent
pythoncom.CoInitialize()
word=win32com.client.DispatchEx('Word.Application')
word.Visible=False;word.DisplayAlerts=0
report=[]
try:
 for p in sorted((ROOT/'submission').glob('*.docx')):
  out=ROOT/'qa'/p.stem;out.mkdir(exist_ok=True)
  print('Rendering',p.name,flush=True)
  doc=word.Documents.Open(str(p.resolve()),ReadOnly=True,AddToRecentFiles=False)
  doc.Repaginate();pdf=out/(p.stem+'.pdf');doc.ExportAsFixedFormat(str(pdf.resolve()),17);doc.Close(False)
  data=pymupdf.open(pdf)
  for i,page in enumerate(data):page.get_pixmap(matrix=pymupdf.Matrix(1.65,1.65),alpha=False).save(out/f'page-{i+1}.png')
  for start in range(0,len(data),6):
   contact=Image.new('RGB',(1200,1700),'#c0c0c0');draw=ImageDraw.Draw(contact)
   for j,i in enumerate(range(start,min(start+6,len(data)))):
    im=Image.open(out/f'page-{i+1}.png');im.thumbnail((580,520));x=(j%2)*600+(600-im.width)//2;y=(j//2)*560+25
    contact.paste(im,(x,y));draw.text(((j%2)*600+12,(j//2)*560+6),f'Page {i+1}',fill='black')
   contact.save(out/f'contact-{start+1}.png')
  (out/'extracted_text.txt').write_text('\n'.join(p.get_text() for p in data),encoding='utf-8')
  report.append({'document':p.name,'pages':len(data),'renderer':'Microsoft Word native PDF export and PyMuPDF PNG render','png_scale':1.65})
  print(p.name,len(data),'pages',flush=True)
finally:
 word.Quit();pythoncom.CoUninitialize()
(ROOT/'qa/render_report.json').write_text(json.dumps(report,indent=2))
