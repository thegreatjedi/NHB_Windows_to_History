import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from shutil import copyfileobj
from typing import Dict, Set

from fastapi import Depends, FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from database import crud, models
from database.database import SessionLocal, engine
from nlp.attn_captioning import generate_caption, get_similar_words
from obj_detect.detector import detect_labels, init_detector


class Context:
    templates = Jinja2Templates(directory='templates')
    detect_func = None
    cat_idx = None
    jobs_done = 0
    total_jobs = 0


models.Base.metadata.create_all(bind=engine)
annotation_executor = ThreadPoolExecutor(max_workers=os.cpu_count()//2)
db_executor = ThreadPoolExecutor(max_workers=1)

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')


# Dependency
async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event('startup')
def startup():
    # Initialisation of object detection components
    Context.detect_func, Context.cat_idx = init_detector()
    
    # Pass a dummy image during startup to fully load models, to avoid
    # errors that can otherwise occur if the very first frontend
    # submission consists of multiple images
    filepath = './dummy.png'
    get_similar_words(
        list(generate_caption(filepath)),
        detect_labels(filepath, Context.detect_func, Context.cat_idx))


@app.on_event('shutdown')
def shutdown():
    annotation_executor.shutdown()
    db_executor.shutdown()


def db_update(db: Session, filename: str, caption: str, tags: Set[str]):
    print(f'Updating {filename} to database.')
    crud.create_photo(db, filename, caption, tags)


def annotate_image(filename: str, filepath: str, db: SessionLocal):
    print(f'{filename}: Beginning annotation.')
    captions = deque()
    for i in range(30):
        captions.append(generate_caption(filepath))
    captions = list(captions)
    print(f'{filename}: Captions generated, beginning object detection.')
    labels = detect_labels(filepath, Context.detect_func, Context.cat_idx)
    words = get_similar_words(captions, labels)
    db_executor.submit(db_update, db, filename, captions[0], words)
    Context.jobs_done = Context.jobs_done + 1
    print(f'{filename} annotated. '
          f'({Context.jobs_done} / {Context.total_jobs} completed)')


@app.get('/', response_class=HTMLResponse)
async def root(request: Request) -> Context.templates.TemplateResponse:
    return Context.templates.TemplateResponse('index.html', {
        'request': request
    })


@app.post('/api/uploadFile')
async def upload_file(
        file: UploadFile = Form(...), db: Session = Depends(get_db)) -> Dict:
    if file.content_type.startswith('image/'):
        filepath = f'./static/uploads/{file.filename}'
        with open(filepath, 'wb') as buffer:
            copyfileobj(file.file, buffer)
            Context.total_jobs = Context.total_jobs + 1
            print(f'{file.filename}: Queuing job. '
                  f'(Total jobs: {Context.total_jobs}')
            annotation_executor.submit(
                annotate_image, file.filename, filepath, db)
    return {
        'uploadSuccess': True
    }
