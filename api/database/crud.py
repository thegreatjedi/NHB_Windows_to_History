from collections import deque
from typing import List, Set

from sqlalchemy.orm import Session

from . import models


def convert_tag_set(tags: Set[str]) -> str:
    return ','.join([tag.replace(',', '\\,') for tag in tags])


def convert_tag_str(tags: str) -> Set[str]:
    part_list = tags.split(',')
    tag_deque = deque(part_list[0:1])
    for tag_part in part_list[1:]:
        if tag_deque[-1].endswith('\\'):
            tag_deque[-1] = f'{tag_deque[-1]},{tag_part}'
        else:
            tag_deque.append(tag_part)
    
    return set(tag_deque)


def create_photo(db: Session, filename: str, caption: str, tags: Set[str]):
    db_photo = models.Photo(
        filename=filename, caption=caption, tags=convert_tag_set(tags))
    db.add(db_photo)
    db.commit()
    db.refresh(db_photo)


def get_photo(db: Session, filename: str) -> models.Photo:
    return db.query(models.Photo).filter(models.Photo.filename == filename)\
        .first()


def get_all_photos(db: Session) -> List[models.Photo]:
    return db.query(models.Photo).all()
