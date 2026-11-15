"""Read scoring CSVs without an audit database; map dataset roots for local review."""
import csv
import io
import json
import math
import posixpath
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image
from quality_paths import map_stored_path


class ScoreView:
    def __init__(self, directory, mappings=()):
        self.directory=Path(directory).resolve()
        self.mappings=mappings
        self.config=json.loads((self.directory/'评分配置.json').read_text(encoding='utf-8-sig'))
        with (self.directory/'逐图评分.csv').open(encoding='utf-8-sig',newline='') as stream:
            self.rows=list(csv.DictReader(stream))
        self.fields=['综合质量分','图像质量分','a1曝光亮度','a2整体清晰度','a3文字水印遮挡',
                     'a4目标区域占比','b整体标注质量','b1类别准确性','b2区域一致性']
        self.classes=sorted({c for row in self.rows for c in row['类别'].split(';') if c})

    @staticmethod
    def number(value):
        try:
            value=float(value)
            return value if math.isfinite(value) else None
        except (ValueError,TypeError):
            return None

    def query(self, args):
        field=args.get('field','综合质量分')
        if field not in self.fields: raise ValueError('未知评分项')
        low=float(args.get('min') or 0);high=float(args.get('max') or 5)
        coverage=float(args.get('coverage') or 0)
        if not 0<=low<=high<=5 or not 0<=coverage<=100: raise ValueError('分数范围为0～5，覆盖率为0～100')
        category=args.get('class','');query=args.get('query','').casefold()
        selected=[]
        for index,row in enumerate(self.rows):
            if category and category not in row['类别'].split(';'): continue
            if query and query not in (row['图片路径']+' '+row['原文件名']).casefold(): continue
            value=self.number(row.get(field));covered=self.number(row.get('评分覆盖率(%)')) or 0
            if value is None or not low<=value<=high or covered<coverage: continue
            selected.append((index,row,value))
        selected.sort(key=lambda item:(item[2],item[0]),reverse=args.get('order')=='desc')
        page=max(0,int(args.get('page',0)));start=page*24
        return {'total':len(selected),'page':page,
                'mean':round(sum(x[2] for x in selected)/len(selected),3) if selected else None,
                'items':[{'id':index,**row} for index,row,_ in selected[start:start+24]]}

    def image_path(self,index):
        row=self.rows[index]
        value=row['图片路径'].replace('\\','/')
        # CSV paths are relative to the original result folder; preserve that root for mapping.
        output=self.config['output'].replace('\\','/')
        original=posixpath.normpath(posixpath.join(output,value))
        if self.mappings: return map_stored_path(original,self.mappings)
        local=self.directory/value
        return local.resolve() if local.is_file() else map_stored_path(original)

    def image(self,index):
        with Image.open(self.image_path(index)) as im:
            im.thumbnail((1000,1000))
            stream=io.BytesIO();im.convert('RGB').save(stream,format='JPEG',quality=85)
            return stream.getvalue()

    def boxes(self,index):
        image=self.image_path(index)
        parts=list(image.parts);position=max(i for i,p in enumerate(parts) if p=='images')
        parts[position]='annotations'
        xml=Path(*parts).with_suffix('.xml')
        root=ET.parse(xml).getroot()
        width=float(root.findtext('size/width'));height=float(root.findtext('size/height'))
        result=[]
        for obj in root.findall('object'):
            coords=[float(obj.findtext('bndbox/'+key)) for key in ('xmin','ymin','xmax','ymax')]
            result.append({'class':obj.findtext('name'),
                           'coords':[v/(width if i%2==0 else height) for i,v in enumerate(coords)]})
        return result


def serve(directory,mappings=(),port=8765):
    view=ScoreView(directory,mappings)
    asset=Path(__file__).resolve().parents[1]/'quality_review'/'scores.html'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass

        def send(self,data,mime='application/json; charset=utf-8',status=200):
            if not isinstance(data,bytes): data=json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
            self.end_headers();self.wfile.write(data)

        def do_GET(self):
            request=urlsplit(self.path);args={k:v[0] for k,v in parse_qs(request.query).items()}
            try:
                if request.path=='/': self.send(asset.read_bytes(),'text/html; charset=utf-8')
                elif request.path=='/api/meta':
                    self.send({'classes':view.classes,'fields':view.fields,'count':len(view.rows),'directory':str(directory)})
                elif request.path=='/api/samples': self.send(view.query(args))
                elif request.path in ('/api/image','/api/boxes'):
                    index=int(args['id'])
                    if not 0<=index<len(view.rows): raise ValueError('未知图片')
                    if request.path=='/api/image': self.send(view.image(index),'image/jpeg')
                    else: self.send(view.boxes(index))
                else: self.send({'error':'地址不存在'},status=404)
            except (OSError,ET.ParseError):
                self.send({'error':'图片/XML未找到；请同步原名数据集，并用 --path-map 映射数据集根目录。'},status=404)
            except (ValueError,KeyError,IndexError,TypeError) as exc:
                self.send({'error':str(exc)},status=400)
            except (BrokenPipeError,ConnectionResetError): pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'综合评分网页 http://127.0.0.1:{server.server_port}\n评分目录 {directory}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
    return 0
