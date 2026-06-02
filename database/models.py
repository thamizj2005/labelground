import os
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, Text, DateTime, ForeignKey, Enum as SQLEnum, JSON, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from sqlalchemy.exc import IntegrityError
import json

Base = declarative_base()

class AnnotationType(str, Enum):
    BBOX = "bbox"
    POLYGON = "polygon"
    KEYPOINTS = "keypoints"

class UserRole(str, Enum):
    ADMIN = "admin"
    SUB_ADMIN = "sub_admin"
    ANNOTATOR = "annotator"
    REVIEWER = "reviewer"

class ImageStatus(str, Enum):
    PENDING = "pending"
    PROCESSED = "processed"
    ERROR = "error"

class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    NEEDS_EDIT = "needs_edit"

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    annotation_type = Column(SQLEnum(AnnotationType), nullable=False)
    label_schema = Column(JSON, nullable=False)  # {classes: [{id, name}], keypoints: []}
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True) # Nullable for legacy/system projects
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    owner = relationship("User", back_populates="owned_projects")
    images = relationship("Image", back_populates="project", cascade="all, delete-orphan")
    assignments = relationship("ProjectAssignment", back_populates="project", cascade="all, delete-orphan")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Validate annotation_type cannot be changed after creation
        self._original_annotation_type = self.annotation_type
    
    def validate_annotation_data(self, data: Dict) -> bool:
        """Validate annotation data matches project type"""
        if self.annotation_type == AnnotationType.BBOX:
            return self._validate_bbox(data)
        elif self.annotation_type == AnnotationType.POLYGON:
            return self._validate_polygon(data)
        elif self.annotation_type == AnnotationType.KEYPOINTS:
            return self._validate_keypoints(data)
        return False
    
    def _validate_bbox(self, data: Dict) -> bool:
        required = {"x", "y", "width", "height", "class_id"}
        if not all(k in data for k in required):
            return False
        # Validate coordinates are normalized (0-1) with small epsilon for float precision
        eps = 0.0001
        for key in ["x", "y", "width", "height"]:
            if not -eps <= data[key] <= 1 + eps:
                return False
        return True
    
    def _validate_polygon(self, data: Dict) -> bool:
        if "points" not in data or "class_id" not in data:
            return False
        if not isinstance(data["points"], list):
            return False
        # Validate each point has x,y and are normalized with small epsilon
        eps = 0.0001
        for point in data["points"]:
            if not isinstance(point, dict) or not all(k in point for k in ["x", "y"]):
                return False
            if not (-eps <= point["x"] <= 1 + eps and -eps <= point["y"] <= 1 + eps):
                return False
        return True
    
    def _validate_keypoints(self, data: Dict) -> bool:
        if "points" not in data or "class_id" not in data:
            return False
        if not isinstance(data["points"], list):
            return False
        # Validate against project's keypoint definitions
        keypoint_defs = self.label_schema.get("keypoints", [])
        if len(data["points"]) != len(keypoint_defs):
            return False
        # Each point can be None (not annotated) or have x,y,visible
        for point in data["points"]:
            if point is not None:
                if not isinstance(point, dict):
                    return False
                if not all(k in point for k in ["x", "y", "visible"]):
                    return False
                eps = 0.0001
                if not (-eps <= point["x"] <= 1 + eps and -eps <= point["y"] <= 1 + eps):
                    return False
                if not isinstance(point["visible"], bool):
                    return False
        return True

class Image(Base):
    __tablename__ = "images"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False, unique=True)
    width = Column(Integer)
    height = Column(Integer)
    status = Column(SQLEnum(ImageStatus), default=ImageStatus.PENDING)
    verification_status = Column(String, default="unverified")  # unverified, verified, needs_edit
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="images")
    annotations = relationship("Annotation", back_populates="image", cascade="all, delete-orphan")
    
    @property
    def latest_annotation(self):
        """Get the latest annotation version"""
        return max(self.annotations, key=lambda a: a.version, default=None)

class Annotation(Base):
    __tablename__ = "annotations"
    
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False, index=True)
    version = Column(Integer, nullable=False, default=1)
    data = Column(JSON, nullable=False)  # List of annotation objects
    created_by = Column(String, nullable=False)  # 'human' or 'auto'
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True) # The actual user who made this
    locked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    image = relationship("Image", back_populates="annotations")
    user = relationship("User", back_populates="annotations")
    
    __table_args__ = (UniqueConstraint('image_id', 'version', name='_image_version_uc'),)

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.ANNOTATOR)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    security_question = Column(String, nullable=True)
    security_answer = Column(String, nullable=True)
    
    # Relationships
    owned_projects = relationship("Project", back_populates="owner")
    annotations = relationship("Annotation", back_populates="user")
    assignments = relationship("ProjectAssignment", back_populates="user", cascade="all, delete-orphan")
    created_users = relationship("User", remote_side=[id], backref="creator")

class ActivityLog(Base):
    __tablename__ = "activity_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    action = Column(String, nullable=False) # login, logout, create_project, delete_project, save_annotation, system_error, etc.
    details = Column(JSON, nullable=True)
    level = Column(String, default="info")  # info, warning, error
    traceback = Column(Text, nullable=True)  # For system error stack traces
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    project = relationship("Project")

class ProjectAssignment(Base):
    __tablename__ = "project_assignments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    can_edit = Column(Boolean, default=True)
    can_export = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User", back_populates="assignments")
    project = relationship("Project", back_populates="assignments")
    
    __table_args__ = (UniqueConstraint('user_id', 'project_id', name='_user_project_uc'),)

# Initialize database
def init_database(db_path: str = "workspace/meta.db"):
    """Initialize database with proper schema"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    return engine

# Session factory
def get_session_factory(db_path: str = "workspace/meta.db"):
    """Get session factory for database"""
    engine = create_engine(f"sqlite:///{db_path}")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal

# Global session factory
SessionLocal = None

def init_session_factory(db_path: str = "workspace/meta.db"):
    """Initialize global session factory"""
    global SessionLocal
    engine = create_engine(f"sqlite:///{db_path}")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal
