from shutil import copyfileobj
from threading import Thread
from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from obj_detect.detector import init_detector, detect_labels
from nlp.attn_captioning import generate_caption, get_similar_words
from traceback import print_exc
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import List


class Context:
    templates = Jinja2Templates(directory='templates')
    detect_func = None
    cat_idx = None


db_executor = ThreadPoolExecutor(max_workers=1)
app = FastAPI()

app.mount('/static', StaticFiles(directory='static'), name='static')


@app.on_event('startup')
def startup():
    Context.detect_func, Context.cat_idx = init_detector()
    generate_caption('./dummy.png')


@app.on_event('shutdown')
def shutdown():
    db_executor.shutdown()


@app.get('/', response_class=HTMLResponse)
async def root(request: Request):
    return Context.templates.TemplateResponse('index.html', {
        'request': request
    })


@app.post('/api/uploadFile')
async def upload_file(file: UploadFile = Form(...)):
    if file.content_type.startswith('image/'):
        filepath = f'./static/uploads/{file.filename}'
        with open(filepath, 'wb') as buffer:
            copyfileobj(file.file, buffer)
            Thread(
                target=annotate_image, name=f'{filepath}_annotation_thread',
                args=[file.filename, filepath]).start()
    return {
        'uploadSuccess': True
    }


def db_update(filename: str, caption: str, tags: List[str]):
    print(f'{filename}: {caption}\n'
          f'Tags: {tags}')


def annotate_image(filename: str, filepath: str):
    print(f'Beginning annotation for {filename}')
    captions = deque()
    for i in range(30):
        captions.append(generate_caption(filepath))
    labels = detect_labels(filepath, Context.detect_func, Context.cat_idx)
    words = get_similar_words(captions, labels)
    try:
        db_executor.submit(db_update, filename, captions[0], words)
    except RuntimeError:
        print_exc()
        print(f'{filename} failed')
