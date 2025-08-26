from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, JSON, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    sessions = relationship("Session", back_populates="user")

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(500))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    token_count = Column(Integer)
    response_time_ms = Column(Integer)
    
    session = relationship("Session", back_populates="messages")
    sources = relationship("MessageSource", back_populates="message")

class DocumentSource(Base):
    __tablename__ = "document_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    source_type = Column(String(100), nullable=False)
    file_path = Column(Text)
    url = Column(Text)
    extra_metadata = Column(JSON)    # renamed from metadata
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    message_sources = relationship("MessageSource", back_populates="document_source")


class MessageSource(Base):
    __tablename__ = "message_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False)
    document_source_id = Column(Integer, ForeignKey("document_sources.id"), nullable=False)
    relevance_score = Column(DECIMAL(5,4))
    chunk_text = Column(Text)
    
    message = relationship("Message", back_populates="sources")
    document_source = relationship("DocumentSource", back_populates="message_sources")
