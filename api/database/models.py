from sqlalchemy import Column, Integer, String

from .database import Base


class Photo(Base):
    __tablename__ = 'photos'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String, index=True, nullable=False, unique=True)
    caption = Column(String)
    tags = Column(String)
