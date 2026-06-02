from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from typing import List, Optional, Dict
import os
from pathlib import Path
import uuid
import asyncio
import shutil
import hashlib
from datetime import datetime

from database.models import init_database, init_session_factory, Project, Image, Annotation, AnnotationType, ImageStatus, ActivityLog
import database.models as db_models

def log_activity(db: Session, user_id: int, action: str, project_id: Optional[int] = None, details: Optional[Dict] = None, level: str = "info", traceback_str: Optional[str] = None):
    """Helper to log user actions"""
    try:
        log = ActivityLog(
            user_id=user_id,
            project_id=project_id,
            action=action,
            details=details,
            level=level,
            traceback=traceback_str
        )
        db.add(log)
        db.commit()
    except Exception as e:
        print(f"Failed to log activity: {e}")

def log_system_error(db: Session, action: str, error_msg: str, traceback_str: str = None, project_id: Optional[int] = None):
    """Log system errors without requiring a user context (user_id=0 for system)"""
    try:
        # Use user_id=0 as a sentinel for system-generated logs
        log = ActivityLog(
            user_id=0,
            project_id=project_id,
            action=action,
            details={"error": error_msg},
            level="error",
            traceback=traceback_str
        )
        db.add(log)
        db.commit()
    except Exception as e:
        print(f"Failed to log system error: {e}")
from filesystem.workspace import ProjectWorkspace, VideoProcessor, ImageImporter, WorkspaceError
from pydantic import BaseModel, validator, ValidationError
import json
import torch
import logging
from backend.export import ExportManager, ExportConfig, AugmentationConfig
from backend.auth import (
    get_password_hash, 
    verify_password, 
    create_access_token, 
    decode_access_token
)
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# Configure logging
logger = logging.getLogger(__name__)

# Pydantic Models for API

# Global Task Registry
TASKS = {}
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

class UserRegister(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserOut(BaseModel):
    id: int
    username: str
    email: str
    role: str
    created_at: datetime

class ProjectCreate(BaseModel):
    name: str
    annotation_type: AnnotationType
    label_schema: dict
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Project name cannot be empty')
        if len(v) > 100:
            raise ValueError('Project name too long')
        # Remove invalid characters
        v = ''.join(c for c in v if c.isalnum() or c in ' -_')
        return v.strip()

class ImageImportRequest(BaseModel):
    paths: List[str]
    is_folder: bool = False

class VideoImportRequest(BaseModel):
    video_path: str
    fps: float = 1.0
    
    @validator('fps')
    def validate_fps(cls, v):
        if v <= 0 or v > 60:
            raise ValueError('FPS must be between 0.1 and 60')
        return v

class AnnotationSaveRequest(BaseModel):
    data: List[dict]
    created_by: str = "human"
    
    @validator('created_by')
    def validate_created_by(cls, v):
        if v not in ["human", "auto"]:
            raise ValueError('created_by must be "human" or "auto"')
        return v

class AnnotationLockRequest(BaseModel):
    locked: bool = True

class ClassAddRequest(BaseModel):
    name: str
    color: str
    prompt: Optional[str] = None

class ClassUpdateRequest(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None
    prompt: Optional[str] = None

class SegmentationRequest(BaseModel):
    x: float
    y: float
    width: Optional[float] = None
    height: Optional[float] = None
    points: Optional[List[Dict[str, float]]] = None # For multi-point prompts

# FastAPI App
app = FastAPI(title="Offline Annotation Platform", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for offline use
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
def get_db():
    db = db_models.SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
    user = db.query(db_models.User).filter(db_models.User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# Initialize workspace
WORKSPACE_PATH = Path("workspace").resolve()
WORKSPACE_PATH.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = Path("static/favicon.ico")
    if favicon_path.exists():
        return FileResponse(favicon_path)
    return Response(status_code=204) # No content

@app.on_event("startup")
async def startup():
    """Initialize database on startup"""
    init_database(str(WORKSPACE_PATH / "meta.db"))
    init_session_factory(str(WORKSPACE_PATH / "meta.db"))
    
    # Create or update default admin
    db = db_models.SessionLocal()
    admin = db.query(db_models.User).filter(db_models.User.username == "admin").first()
    if not admin:
        admin = db_models.User(
            username="admin",
            email="admin@labelground.ai",
            hashed_password=get_password_hash("admin123"),
            role=db_models.UserRole.ADMIN,
            security_question="What is your favorite color?",
            security_answer=get_password_hash("blue")
        )
        db.add(admin)
        db.commit()
    elif not admin.security_question:
        # Retrofit existing admin with default question
        admin.security_question = "What is your favorite color?"
        admin.security_answer = get_password_hash("blue")
        db.commit()
    db.close()

# API Endpoints
@app.post("/api/auth/register", response_model=UserOut)
def register_user(user: UserRegister, db: Session = Depends(get_db)):
    email = f"{user.username}@labelground.local"
    # Legacy open registration defaults to ANNOTATOR without a creator
    if db.query(db_models.User).filter(db_models.User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    
    db_user = db_models.User(
        username=user.username,
        email=email,
        hashed_password=get_password_hash(user.password),
        role=db_models.UserRole.ANNOTATOR
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

class UserAdminCreate(BaseModel):
    username: str
    password: str
    role: db_models.UserRole

@app.post("/api/users", response_model=UserOut)
def create_user_by_admin(
    user: UserAdminCreate, 
    current_user: db_models.User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """Admin and Sub-Admin can create users"""
    if current_user.role not in [db_models.UserRole.ADMIN, db_models.UserRole.SUB_ADMIN]:
        raise HTTPException(status_code=403, detail="Not authorized to create users")
        
    if current_user.role == db_models.UserRole.SUB_ADMIN and user.role != db_models.UserRole.ANNOTATOR:
        raise HTTPException(status_code=403, detail="Sub-administrators can only create Annotators")

    email = f"{user.username}@labelground.local"

    if db.query(db_models.User).filter(db_models.User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    
    db_user = db_models.User(
        username=user.username,
        email=email,
        hashed_password=get_password_hash(user.password),
        role=user.role,
        created_by_id=current_user.id
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/api/users", response_model=List[UserOut])
def get_users(
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List users for admin management"""
    if current_user.role == db_models.UserRole.ADMIN:
        return db.query(db_models.User).all()
    elif current_user.role == db_models.UserRole.SUB_ADMIN:
        # Returns themselves and users they've explicitly created
        users = db.query(db_models.User).filter(
            (db_models.User.id == current_user.id) | 
            (db_models.User.created_by_id == current_user.id)
        ).all()
        return users
    else:
        # Regular users can only see themselves
        return [current_user]

class UserAdminEdit(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    role: Optional[db_models.UserRole] = None

@app.patch("/api/users/{target_id}", response_model=UserOut)
def update_user_by_admin(
    target_id: int,
    req: UserAdminEdit,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    target_user = db.query(db_models.User).filter(db_models.User.id == target_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Authorization checks
    if current_user.role == db_models.UserRole.ADMIN:
        # Admin can edit sub-admins and annotators (or themselves)
        if target_user.role == db_models.UserRole.ADMIN and target_user.id != current_user.id:
            raise HTTPException(status_code=403, detail="Cannot edit another Admin")
    elif current_user.role == db_models.UserRole.SUB_ADMIN:
        # Sub admin can only edit annotators they created
        if target_user.id != current_user.id and (target_user.role != db_models.UserRole.ANNOTATOR or target_user.created_by_id != current_user.id):
             raise HTTPException(status_code=403, detail="Not authorized to edit this user")
    else:
        raise HTTPException(status_code=403, detail="Only administrators can edit users")

    if req.username and req.username != target_user.username:
        if db.query(db_models.User).filter(db_models.User.username == req.username).first():
            raise HTTPException(status_code=400, detail="Username already in use")
        target_user.username = req.username
        
    if req.role and target_user.id != current_user.id:
         target_user.role = req.role

    if req.password:
        if len(req.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
        target_user.hashed_password = get_password_hash(req.password)

    db.commit()
    db.refresh(target_user)
    return target_user

@app.post("/api/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(db_models.User).filter(db_models.User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user.username})
    
    # Log activity
    log_activity(db, user.id, "login")
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/auth/me", response_model=UserOut)
def get_me(current_user: db_models.User = Depends(get_current_user)):
    return current_user

# -----------------------------------------------
# Password Reset Endpoints
# -----------------------------------------------

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

class AdminResetPasswordRequest(BaseModel):
    username: str
    new_password: str

@app.post("/api/auth/change-password")
def change_password(
    req: ChangePasswordRequest,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lets a logged-in user change their own password after providing the old one."""
    if not verify_password(req.old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Old password is incorrect")
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")
    current_user.hashed_password = get_password_hash(req.new_password)
    db.commit()
    return {"message": "Password changed successfully"}

@app.post("/api/auth/admin-reset-password")
def admin_reset_password(
    req: AdminResetPasswordRequest,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Admin-only: reset any user's password."""
    if current_user.role != db_models.UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    target = db.query(db_models.User).filter(db_models.User.username == req.username).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    target.hashed_password = get_password_hash(req.new_password)
    db.commit()
    return {"message": f"Password for '{req.username}' has been reset"}


class SecurityQuestionReset(BaseModel):
    username: str
    answer: str
    new_password: str

@app.get("/api/auth/security-question")
def get_security_question(username: str, db: Session = Depends(get_db)):
    """Get the user's security question if it exists."""
    user = db.query(db_models.User).filter(db_models.User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.security_question:
        raise HTTPException(status_code=400, detail="User has no security question set")
    return {"question": user.security_question}

@app.post("/api/auth/reset-password-security")
def reset_password_security(req: SecurityQuestionReset, db: Session = Depends(get_db)):
    """Reset password via security question answer."""
    user = db.query(db_models.User).filter(db_models.User.username == req.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.security_answer:
        raise HTTPException(status_code=400, detail="User has no security answer set")
        
    try:
        # Check standard hashing
        if not verify_password(req.answer.lower().strip(), user.security_answer):
            raise ValueError()
    except Exception:
        # Fallback to plain-text string match if it was stored without hashing (e.g. earlier implementations or custom updates)
        if user.security_answer.lower().strip() != req.answer.lower().strip():
             raise HTTPException(status_code=400, detail="Incorrect answer")
            
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")
        
    user.hashed_password = get_password_hash(req.new_password)
    db.commit()
    return {"message": "Password reset successfully"}


@app.post("/api/projects", response_model=dict)
def create_project(
    project: ProjectCreate,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create new project with annotation type"""
    
    # Only ADMIN and SUB_ADMIN can create projects
    if current_user.role not in [db_models.UserRole.ADMIN, db_models.UserRole.SUB_ADMIN]:
        raise HTTPException(status_code=403, detail="Only administrators can create projects")
    
    # Check if project exists (case-insensitive)
    existing = db.query(Project).filter(func.lower(Project.name) == func.lower(project.name)).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Project with name '{project.name}' already exists")
    
    # Validate label_schema based on annotation type
    if project.annotation_type == AnnotationType.KEYPOINTS:
        if "keypoints" not in project.label_schema:
            raise HTTPException(status_code=400, detail="Keypoint projects require keypoint definitions")
    
    # Create project in database
    db_project = Project(
        name=project.name,
        annotation_type=project.annotation_type,
        label_schema=project.label_schema,
        owner_id=current_user.id
    )
    db.add(db_project)
    db.flush() # Get ID without committing yet

    # Auto-assign owner as a full member (can_edit + can_export)
    from database.models import ProjectAssignment
    owner_assignment = ProjectAssignment(
        user_id=current_user.id,
        project_id=db_project.id,
        can_edit=True,
        can_export=True
    )
    db.add(owner_assignment)
    
    # Create workspace directories
    workspace = ProjectWorkspace(WORKSPACE_PATH, project.name)
    workspace.create()
    
    # Final commit
    db.commit()
    db.refresh(db_project)
    
    # Log activity
    log_activity(db, current_user.id, "create_project", project_id=db_project.id, details={"name": db_project.name})
    
    return {
        "id": db_project.id,
        "name": db_project.name,
        "annotation_type": db_project.annotation_type,
        "created_at": db_project.created_at.isoformat()
    }

# ─── Project Team Management Endpoints ───────────────────────────────────────

class AssignMemberRequest(BaseModel):
    username: str
    can_edit: bool = True
    can_export: bool = False

@app.get("/api/projects/{project_id}/members")
def list_project_members(
    project_id: int,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all members of a project. Accessible to any project member."""
    from database.models import ProjectAssignment
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check access: owner, admin, or existing member
    is_member = db.query(ProjectAssignment).filter(
        ProjectAssignment.project_id == project_id,
        ProjectAssignment.user_id == current_user.id
    ).first()
    if not is_member and project.owner_id != current_user.id and current_user.role != db_models.UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Not a project member")

    assignments = db.query(ProjectAssignment).filter(ProjectAssignment.project_id == project_id).all()
    return [
        {
            "user_id": a.user_id,
            "username": a.user.username,
            "role": a.user.role,
            "can_edit": a.can_edit,
            "can_export": a.can_export,
            "is_owner": (a.user_id == project.owner_id)
        }
        for a in assignments
    ]

@app.post("/api/projects/{project_id}/members")
def assign_project_member(
    project_id: int,
    req: AssignMemberRequest,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Assign a user to a project. Only project owner or admin can do this."""
    from database.models import ProjectAssignment
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Only owner or admin can assign
    if project.owner_id != current_user.id and current_user.role != db_models.UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Only project owner or admin can assign members")

    # Find target user
    target = db.query(db_models.User).filter(db_models.User.username == req.username).first()
    if not target:
        raise HTTPException(status_code=404, detail=f"User '{req.username}' not found")

    # Check if already assigned
    existing = db.query(ProjectAssignment).filter(
        ProjectAssignment.project_id == project_id,
        ProjectAssignment.user_id == target.id
    ).first()
    if existing:
        # Update permissions instead
        existing.can_edit = req.can_edit
        existing.can_export = req.can_export
        db.commit()
        return {"message": f"Updated permissions for '{req.username}'"}

    assignment = ProjectAssignment(
        user_id=target.id,
        project_id=project_id,
        can_edit=req.can_edit,
        can_export=req.can_export
    )
    db.add(assignment)
    db.commit()
    return {"message": f"User '{req.username}' added to project"}

@app.delete("/api/projects/{project_id}/members/{username}")
def remove_project_member(
    project_id: int,
    username: str,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Remove a user from a project. Only project owner or admin can do this."""
    from database.models import ProjectAssignment
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.owner_id != current_user.id and current_user.role != db_models.UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Only project owner or admin can remove members")

    target = db.query(db_models.User).filter(db_models.User.username == username).first()
    if not target:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")

    # Prevent removing the owner
    if target.id == project.owner_id:
        raise HTTPException(status_code=400, detail="Cannot remove the project owner")

    assignment = db.query(ProjectAssignment).filter(
        ProjectAssignment.project_id == project_id,
        ProjectAssignment.user_id == target.id
    ).first()
    if not assignment:
        raise HTTPException(status_code=404, detail="User is not a member of this project")

    db.delete(assignment)
    db.commit()
    return {"message": f"User '{username}' removed from project"}


@app.get("/api/projects", response_model=List[dict])
def list_projects(
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List projects owned by or assigned to current user"""
    from database.models import ProjectAssignment
    
    if current_user.role == db_models.UserRole.ADMIN:
        all_projects = db.query(Project).all()
    else:
        # Projects owned by user
        owned_projects = db.query(Project).filter(Project.owner_id == current_user.id).all()
        
        # Projects assigned to user
        assigned_projects = db.query(Project).join(ProjectAssignment).filter(ProjectAssignment.user_id == current_user.id).all()
        
        # Merge and deduplicate
        all_projects = list(set(owned_projects + assigned_projects))
    
    return [
        {
            "id": p.id,
            "name": p.name,
            "annotation_type": p.annotation_type,
            "image_count": len(p.images),
            "created_at": p.created_at.isoformat(),
            "role": "owner" if p.owner_id == current_user.id else "annotator"
        }
        for p in all_projects
    ]

@app.get("/api/projects/{project_id}")
def get_project(project_id: int, db: Session = Depends(get_db)):
    """Get project details"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return {
        "id": project.id,
        "name": project.name,
        "annotation_type": project.annotation_type,
        "label_schema": project.label_schema,
        "image_count": len(project.images),
        "created_at": project.created_at.isoformat()
    }

@app.patch("/api/projects/{project_id}/classes")
def add_project_class(
    project_id: int,
    class_req: ClassAddRequest,
    db: Session = Depends(get_db)
):
    """Add a new class to project label schema"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Update label_schema
    schema = project.label_schema.copy()
    if "classes" not in schema:
        schema["classes"] = []
    
    # Check if class name already exists
    if any(c["name"].lower() == class_req.name.lower() for c in schema["classes"]):
        raise HTTPException(status_code=400, detail="Class already exists")
    
    max_id = -1
    for c in schema.get("classes", []):
        if c.get("id", -1) > max_id:
            max_id = c["id"]
    
    new_class = {
        "id": max_id + 1,
        "name": class_req.name,
        "color": class_req.color,
        "prompt": class_req.prompt
    }
    schema["classes"].append(new_class)
    
    project.label_schema = schema
    flag_modified(project, "label_schema")
    db.commit()
    db.refresh(project)
    
    return project.label_schema

@app.patch("/api/projects/{project_id}/classes/{class_id}")
def update_project_class(
    project_id: int,
    class_id: int,
    class_req: ClassUpdateRequest,
    db: Session = Depends(get_db)
):
    """Update an existing class in project label schema"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    schema = project.label_schema.copy()
    classes = schema.get("classes", [])
    
    found = False
    for c in classes:
        if c["id"] == class_id:
            if class_req.name:
                c["name"] = class_req.name
            if class_req.color:
                c["color"] = class_req.color
            if class_req.prompt is not None:
                c["prompt"] = class_req.prompt
            found = True
            break
    
    if not found:
        raise HTTPException(status_code=404, detail="Class not found")
    
    project.label_schema = schema
    flag_modified(project, "label_schema")
    db.commit()
    db.refresh(project)
    
    return project.label_schema

@app.delete("/api/projects/{project_id}/classes/{class_id}")
def delete_project_class(
    project_id: int,
    class_id: int,
    db: Session = Depends(get_db)
):
    """Delete a class from project label schema and its annotations"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    schema = project.label_schema.copy()
    classes = schema.get("classes", [])
    
    # Filter out the class
    new_classes = [c for c in classes if c["id"] != class_id]
    if len(new_classes) == len(classes):
        raise HTTPException(status_code=404, detail="Class not found")
    
    schema["classes"] = new_classes
    project.label_schema = schema
    flag_modified(project, "label_schema")
    
    # Optional: Delete annotations associated with this class?
    # For now, let's keep them but they might be invalid. 
    # Or we can clean them up.
    
    db.commit()
    db.refresh(project)
    return project.label_schema


@app.get("/api/tasks/{task_id}")
def get_task_status(task_id: str):
    """Get status of background task (no auth needed — task IDs are unguessable UUIDs)"""
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.post("/api/projects/{project_id}/images")
def import_images(
    project_id: int,
    import_req: ImageImportRequest,
    background_tasks: BackgroundTasks,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Import images into project"""
    if current_user.role not in [db_models.UserRole.ADMIN, db_models.UserRole.SUB_ADMIN]:
        raise HTTPException(status_code=403, detail="Only administrators can import images")

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    workspace = ProjectWorkspace(WORKSPACE_PATH, project.name)
    
    imported_images = []
    errors = []
    
    for path_str in import_req.paths:
        try:
            source_path = Path(path_str)
            
            if import_req.is_folder:
                # Import folder
                results = ImageImporter.import_folder(source_path, workspace.images_dir)
                for dest_path, img_hash in results:
                    # Check if image already registered in this project
                    existing = db.query(Image).filter(
                        Image.filepath == str(dest_path.relative_to(WORKSPACE_PATH))
                    ).first()
                    
                    if not existing:
                        width, height = ImageImporter.get_image_dimensions(dest_path)
                        
                        db_image = Image(
                            project_id=project_id,
                            filename=dest_path.name,
                            filepath=str(dest_path.relative_to(WORKSPACE_PATH)),
                            width=width,
                            height=height,
                            status=ImageStatus.PROCESSED
                        )
                        db.add(db_image)
                        imported_images.append({
                            "filename": dest_path.name,
                            "path": str(dest_path),
                            "dimensions": {"width": width, "height": height}
                        })
            else:
                # Import single image
                dest_path, img_hash = ImageImporter.import_image(source_path, workspace.images_dir)
                
                # Check if image already registered in this project
                existing = db.query(Image).filter(
                    Image.filepath == str(dest_path.relative_to(WORKSPACE_PATH))
                ).first()
                
                if existing:
                    if existing.project_id != project_id:
                        errors.append(f"Image {source_path.name} already belongs to another project")
                    continue
                
                width, height = ImageImporter.get_image_dimensions(dest_path)
                
                db_image = Image(
                    project_id=project_id,
                    filename=dest_path.name,
                    filepath=str(dest_path.relative_to(WORKSPACE_PATH)),
                    width=width,
                    height=height,
                    status=ImageStatus.PROCESSED
                )
                db.add(db_image)
                logger.info(f"[Project: {project.name}] 📸 Imported image: {dest_path.name} ({width}x{height})")
                imported_images.append({
                    "filename": dest_path.name,
                    "path": str(dest_path),
                    "dimensions": {"width": width, "height": height}
                })
                
        except Exception as e:
            errors.append(f"Error importing {path_str}: {str(e)}")
    
    db.commit()
    
    return {
        "imported_count": len(imported_images),
        "imported": imported_images,
        "errors": errors
    }

@app.post("/api/projects/{project_id}/videos")
async def import_video(
    project_id: int,
    import_req: VideoImportRequest,
    background_tasks: BackgroundTasks,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Import video and extract frames"""
    if current_user.role not in [db_models.UserRole.ADMIN, db_models.UserRole.SUB_ADMIN]:
        raise HTTPException(status_code=403, detail="Only administrators can import videos")

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    video_path = Path(import_req.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=400, detail="Video file not found")
    
    workspace = ProjectWorkspace(WORKSPACE_PATH, project.name)
    
    # Copy video to project raw_videos directory
    video_dest = workspace.raw_videos_dir / video_path.name
    shutil.copy2(video_path, video_dest)
    
    # Estimate total frames
    try:
        estimated_frames, video_fps = VideoProcessor.estimate_total_frames(
            video_dest, import_req.fps
        )
    except WorkspaceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Start background extraction
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "status": "processing",
        "progress": 0,
        "total": estimated_frames,
        "current": 0
    }
    
    def extract_frames_task():
        try:
            def update_progress(count):
                TASKS[task_id]["current"] = count
                TASKS[task_id]["progress"] = int((count / estimated_frames) * 100) if estimated_frames > 0 else 0
            
            # Calculate video hash for unique frame naming
            video_hash = ImageImporter.calculate_image_hash(video_dest)
            
            frame_paths = VideoProcessor.extract_frames(
                video_dest,
                workspace.images_dir,
                import_req.fps,
                progress_callback=update_progress,
                video_hash=video_hash
            )
            
            # Register frames in database
            db_local = db_models.SessionLocal()
            try:
                for frame_path in frame_paths:
                    filepath_rel = str(frame_path.relative_to(WORKSPACE_PATH))
                    
                    # Check if already exists in DB
                    existing = db_local.query(Image).filter(Image.filepath == filepath_rel).first()
                    if existing:
                        continue
                        
                    width, height = ImageImporter.get_image_dimensions(frame_path)
                    
                    db_image = Image(
                        project_id=project_id,
                        filename=frame_path.name,
                        filepath=filepath_rel,
                        width=width,
                        height=height,
                        status=ImageStatus.PROCESSED
                    )
                    db_local.add(db_image)
                
                db_local.commit()
                TASKS[task_id]["status"] = "completed"
                TASKS[task_id]["progress"] = 100
            finally:
                db_local.close()
                
        except Exception as e:
            print(f"Video extraction failed: {e}")
            TASKS[task_id]["status"] = "failed"
            TASKS[task_id]["error"] = str(e)
    
    background_tasks.add_task(extract_frames_task)
    
    return {
        "task_id": task_id,
        "video_path": str(video_dest),
        "target_fps": import_req.fps,
        "original_fps": video_fps,
        "status": "processing"
    }


@app.post("/api/projects/{project_id}/upload-images")
async def upload_images(
    project_id: int,
    files: List[UploadFile] = File(...),
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload images directly from browser"""
    if current_user.role not in [db_models.UserRole.ADMIN, db_models.UserRole.SUB_ADMIN]:
        raise HTTPException(status_code=403, detail="Only administrators can upload images")

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    workspace = ProjectWorkspace(WORKSPACE_PATH, project.name)
    imported_images = []
    errors = []
    
    for file in files:
        try:
            # Save uploaded file to temporary location
            temp_path = WORKSPACE_PATH / "temp" / file.filename
            temp_path.parent.mkdir(exist_ok=True)
            
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Import the image
            dest_path, img_hash = ImageImporter.import_image(temp_path, workspace.images_dir)
            
            # Check if already exists
            existing = db.query(Image).filter(
                Image.filepath == str(dest_path.relative_to(WORKSPACE_PATH))
            ).first()
            
            if not existing:
                width, height = ImageImporter.get_image_dimensions(dest_path)
                
                db_image = Image(
                    project_id=project_id,
                    filename=dest_path.name,
                    filepath=str(dest_path.relative_to(WORKSPACE_PATH)),
                    width=width,
                    height=height,
                    status=ImageStatus.PROCESSED
                )
                db.add(db_image)
                imported_images.append({
                    "filename": dest_path.name,
                    "dimensions": {"width": width, "height": height}
                })
            
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
            
        except Exception as e:
            errors.append(f"Error importing {file.filename}: {str(e)}")
    
    db.commit()
    
    return {
        "imported_count": len(imported_images),
        "imported": imported_images,
        "errors": errors
    }

@app.post("/api/projects/{project_id}/upload-video")
async def upload_video(
    project_id: int,
    file: UploadFile = File(...),
    fps: float = Form(1.0),
    background_tasks: BackgroundTasks = None,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload video directly from browser"""
    if current_user.role not in [db_models.UserRole.ADMIN, db_models.UserRole.SUB_ADMIN]:
        raise HTTPException(status_code=403, detail="Only administrators can upload video")

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    workspace = ProjectWorkspace(WORKSPACE_PATH, project.name)
    
    # Save uploaded video
    video_dest = workspace.raw_videos_dir / file.filename
    with open(video_dest, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Estimate total frames
    try:
        estimated_frames, video_fps = VideoProcessor.estimate_total_frames(
            video_dest, fps
        )
    except WorkspaceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Start background extraction
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "status": "processing",
        "progress": 0,
        "total": estimated_frames,
        "current": 0
    }
    
    def extract_frames_task():
        try:
            def update_progress(count):
                TASKS[task_id]["current"] = count
                TASKS[task_id]["progress"] = int((count / estimated_frames) * 100) if estimated_frames > 0 else 0
            
            # Calculate video hash for unique frame naming
            video_hash = ImageImporter.calculate_image_hash(video_dest)
            
            frame_paths = VideoProcessor.extract_frames(
                video_dest,
                workspace.images_dir,
                fps,
                progress_callback=update_progress,
                video_hash=video_hash
            )
            
            # Register frames in database
            db_local = db_models.SessionLocal()
            try:
                for frame_path in frame_paths:
                    filepath_rel = str(frame_path.relative_to(WORKSPACE_PATH))
                    
                    # Check if already exists in DB
                    existing = db_local.query(Image).filter(Image.filepath == filepath_rel).first()
                    if existing:
                        continue
                        
                    width, height = ImageImporter.get_image_dimensions(frame_path)
                    
                    db_image = Image(
                        project_id=project_id,
                        filename=frame_path.name,
                        filepath=filepath_rel,
                        width=width,
                        height=height,
                        status=ImageStatus.PROCESSED
                    )
                    db_local.add(db_image)
                
                db_local.commit()
                TASKS[task_id]["status"] = "completed"
                TASKS[task_id]["progress"] = 100
            finally:
                db_local.close()
                
        except Exception as e:
            print(f"Video extraction failed: {e}")
            TASKS[task_id]["status"] = "failed"
            TASKS[task_id]["error"] = str(e)
    
    background_tasks.add_task(extract_frames_task)
    
    return {
        "task_id": task_id,
        "video_path": str(video_dest),
        "target_fps": fps,
        "original_fps": video_fps,
        "status": "processing"
    }

@app.get("/api/projects/{project_id}/images")
def list_project_images(
    project_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    sort: str = Query("id", regex="^(id|confidence|created_at)$"),
    db: Session = Depends(get_db)
):
    """List images in project with pagination"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    def get_avg_confidence(img):
        if not img.latest_annotation or not img.latest_annotation.data:
            return 1.0 # High confidence if not annotated or human confirmed
        
        # Human annotations are always 100% certain in this logic
        if img.latest_annotation.created_by == 'human':
            return 1.0
            
        confs = [ann.get('confidence', 0.5) for ann in img.latest_annotation.data if isinstance(ann, dict)]
        if not confs:
            return 0.5 # Default for empty AI output
        return sum(confs) / len(confs)

    query = db.query(Image).filter(Image.project_id == project_id)
    
    if sort == "id":
        images = query.order_by(Image.id.asc()).offset(skip).limit(limit).all()
    else:
        # Sort in memory for confidence/logic since it's not a direct column
        # For large datasets, this should be a column. For offline/small project, memory is fine.
        all_images = query.all()
        if sort == "confidence":
            all_images.sort(key=get_avg_confidence) # Ascending: Low confidence (confusion) first
        elif sort == "created_at":
            all_images.sort(key=lambda x: x.created_at, reverse=True)
            
        images = all_images[skip : skip + limit]
    
    total = query.count()
    
    return {
        "images": [
            {
                "id": img.id,
                "filename": img.filename,
                "dimensions": {"width": img.width, "height": img.height},
                "status": img.status,
                "verification_status": img.verification_status,
                "created_at": img.created_at.isoformat(),
                "has_annotations": len(img.annotations) > 0,
                "latest_version": img.latest_annotation.version if img.latest_annotation else 0,
                "latest_created_by": img.latest_annotation.created_by if img.latest_annotation else None,
                "confidence": get_avg_confidence(img)
            }
            for img in images
        ],
        "total": total,
        "skip": skip,
        "limit": limit
    }

@app.get("/api/images/{image_id}/file")
def serve_image_file(image_id: int, db: Session = Depends(get_db)):
    """Serve image file from workspace"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    image_path = WORKSPACE_PATH / image.filepath
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    
    import mimetypes
    media_type, _ = mimetypes.guess_type(image_path)
    if not media_type:
        # Better fallback for JPEG
        if image_path.suffix.lower() in [".jpg", ".jpeg"]:
            media_type = "image/jpeg"
        else:
            media_type = f"image/{image_path.suffix[1:]}" if image_path.suffix else "image/jpeg"
    
    return FileResponse(
        image_path,
        media_type=media_type
    )

@app.delete("/api/images/{image_id}")
def delete_image(image_id: int, db: Session = Depends(get_db)):
    """Delete image and its files"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # 1. Delete file on disk
    try:
        image_path = WORKSPACE_PATH / image.filepath
        if image_path.exists():
            os.remove(image_path)
    except Exception as e:
        print(f"Error deleting image file: {e}")
        # Continue to delete DB record
        
    # 2. Delete DB record (Cascades to annotations)
    db.delete(image)
    db.commit()
    
    return {"message": "Image deleted successfully"}

@app.patch("/api/images/{image_id}/status")
def update_image_status(image_id: int, status_req: dict, db: Session = Depends(get_db)):
    """Manually update image status (e.g. revert to pending)"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    new_status = status_req.get("status")
    if not new_status:
        raise HTTPException(status_code=400, detail="Status is required")
    
    try:
        new_status_enum = ImageStatus(new_status)
        
        # If reverting to pending, optionally clear all annotations to make it truly "New"
        if new_status_enum == ImageStatus.PENDING:
            if status_req.get("clear_annotations", True):
                db.query(Annotation).filter(Annotation.image_id == image_id).delete()
                
        image.status = new_status_enum
        db.commit()
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {new_status}")
    
    return {"message": f"Status updated to {new_status}", "status": new_status}

@app.post("/api/images/{image_id}/null")
def mark_image_null(image_id: int, db: Session = Depends(get_db)):
    """Mark image as null/negative sample (empty annotations)"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
        
    workspace = ProjectWorkspace(WORKSPACE_PATH, image.project.name)
    
    # Create new version
    version = workspace.get_next_annotation_version(image.filename)
    
    # Save empty data
    try:
        ann_file = workspace.save_annotation(image.filename, version, {
            "version": version,
            "created_by": "human",
            "created_at": datetime.utcnow().isoformat(),
            "data": []  # Empty list for null sample
        })
        
        # Save to DB
        db_ann = Annotation(
            image_id=image_id,
            version=version,
            data=[],
            created_by="human"
        )
        db.add(db_ann)
        
        # Update image status
        image.status = ImageStatus.PROCESSED
        
        db.commit()
        
        return {"message": "Image marked as null", "status": "processed"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/images/{image_id}/annotations")
def save_annotation(
    image_id: int,
    annotation_req: AnnotationSaveRequest,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save new annotation version"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Validate annotation data against project type
    for ann in annotation_req.data:
        if not image.project.validate_annotation_data(ann):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid annotation format for project type {image.project.annotation_type}"
            )
    
    # Get next version number from DB to avoid IntegrityError (with lock to prevent race condition)
    from sqlalchemy import func
    max_version = db.query(func.max(Annotation.version)).filter(
        Annotation.image_id == image_id
    ).with_for_update().scalar()
    next_version = (max_version or 0) + 1
    
    workspace = ProjectWorkspace(WORKSPACE_PATH, image.project.name)
    
    # Check if latest version is locked
    latest_ann = db.query(Annotation).filter(
        Annotation.image_id == image_id,
        Annotation.version == (max_version or 0)
    ).first()
    if latest_ann and latest_ann.locked:
        raise HTTPException(
            status_code=400,
            detail="Latest annotation version is locked"
        )
    
    # Check if this is a correction (user modifying AI annotation)
    is_correction = False
    if latest_ann and latest_ann.created_by == "auto" and annotation_req.created_by == "human" and latest_ann.data:
        # Compare data to see if it's an actual edit or just a confirmation
        if json.dumps(latest_ann.data, sort_keys=True) != json.dumps(annotation_req.data, sort_keys=True):
            is_correction = True
            logger.info(f"📝 Correction detected: User modified AI annotation data for image {image_id}")
        else:
            logger.info(f"✅ Confirmation: User saved AI annotation WITHOUT changes for image {image_id}")
    elif latest_ann and latest_ann.created_by == "auto" and annotation_req.created_by == "human":
        # Fallback if latest_ann.data is missing for some reason
        is_correction = True
        logger.info(f"📝 Correction detected: User modifying AI annotation (missing data for comparison) for image {image_id}")

    
    # Create new annotation in database
    db_annotation = Annotation(
        image_id=image_id,
        version=next_version,
        data=annotation_req.data,
        created_by=annotation_req.created_by,
        user_id=current_user.id if annotation_req.created_by == "human" else None
    )
    db.add(db_annotation)
    
    # Save to filesystem
    ann_file = workspace.save_annotation(
        image.filename,
        next_version,
        annotation_req.data
    )
    
    # Auto-verify when human saves
    if annotation_req.created_by == "human":
        image.verification_status = "verified"
    
    db.commit()
    db.refresh(db_annotation)


    # Log activity
    log_activity(db, current_user.id, "save_annotation", project_id=image.project_id, details={
        "image_id": image_id,
        "version": next_version,
        "is_correction": is_correction,
        "count": len(annotation_req.data)
    })
    
    # Track corrections for fine-tuning
    if is_correction:
        from backend.training_orchestrator import TrainingOrchestrator
        project = image.project
        correction_count = TrainingOrchestrator.increment_correction_count(project, db)
        logger.info(f"[Project: {project.name}] 📝 Correction detected for Image {image_id}. Total corrections = {correction_count}")
    else:
        logger.info(f"[Project: {image.project.name}] ✅ New human annotation saved for Image {image_id} (Version {next_version})")
    
    return {
        "id": db_annotation.id,
        "image_id": image_id,
        "version": next_version,
        "created_by": annotation_req.created_by,
        "created_at": db_annotation.created_at.isoformat(),
        "file_path": str(ann_file.relative_to(WORKSPACE_PATH)),
        "is_correction": is_correction
    }

@app.get("/api/images/{image_id}/annotations")
def get_annotation_versions(
    image_id: int,
    db: Session = Depends(get_db)
):
    """Get all annotation versions for image"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    annotations = db.query(Annotation).filter(
        Annotation.image_id == image_id
    ).order_by(Annotation.version).all()
    
    return [
        {
            "id": ann.id,
            "version": ann.version,
            "created_by": ann.created_by,
            "created_at": ann.created_at.isoformat(),
            "locked": ann.locked,
            "annotation_count": len(ann.data) if ann.data else 0
        }
        for ann in annotations
    ]

@app.get("/api/images/{image_id}/annotations/all")
def get_all_annotations(
    image_id: int,
    db: Session = Depends(get_db)
):
    """Get all annotations from all versions for an image"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    annotations = db.query(Annotation).filter(
        Annotation.image_id == image_id
    ).all()
    
    combined_data = []
    for ann in annotations:
        if ann.data:
            # Tag each annotation with its source version
            version_tagged = [
                {**item, "source_version": ann.version} 
                for item in ann.data
            ]
            combined_data.extend(version_tagged)
            
    return combined_data

@app.get("/api/images/{image_id}/annotations/latest")
def get_latest_annotations(
    image_id: int,
    db: Session = Depends(get_db)
):
    """Get only the latest version of annotations for an image"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Get only the latest annotation version
    latest_annotation = db.query(Annotation).filter(
        Annotation.image_id == image_id
    ).order_by(Annotation.version.desc()).first()
    
    if latest_annotation and latest_annotation.data:
        return latest_annotation.data
    
    return []


class PreviewRequest(BaseModel):
    image_id: int
    augmentation: AugmentationConfig
    grayscale: bool = False

@app.post("/api/preview/augment")
def preview_augmentation(
    req: PreviewRequest,
    db: Session = Depends(get_db)
):
    """Generate a preview of the augmented image"""
    project = db.query(Project).join(Image).filter(Image.id == req.image_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project/Image not found")
        
    manager = ExportManager(WORKSPACE_PATH, project)
    try:
        base64_img = manager.preview_augmentation(req.image_id, req.augmentation, req.grayscale)
        return {"image": base64_img}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects/{project_id}/export")
async def export_project(
    project_id: int,
    config: ExportConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Export project dataset"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # validation
    if config.format not in ["yolo", "coco", "json", "voc", "labelme"]:
        raise HTTPException(status_code=400, detail="Invalid export format")

    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "status": "processing",
        "type": "export",
        "project_id": project_id,
        "progress": 0
    }

    def run_export_task():
        try:
            manager = ExportManager(WORKSPACE_PATH, project)
            TASKS[task_id]["status"] = "processing"
            TASKS[task_id]["message"] = "Exporting dataset..."
            
            output_path = manager.run_export(config)
            
            # Pre-zip immediately so download is instant
            TASKS[task_id]["status"] = "zipping"
            TASKS[task_id]["message"] = "Compressing dataset into zip..."
            TASKS[task_id]["progress"] = 90
            
            export_dir = Path(output_path)
            zip_path = shutil.make_archive(str(export_dir), 'zip', str(export_dir))
            zip_size = os.path.getsize(zip_path)
            
            TASKS[task_id]["status"] = "completed"
            TASKS[task_id]["progress"] = 100
            TASKS[task_id]["output_path"] = output_path
            TASKS[task_id]["zip_path"] = zip_path
            TASKS[task_id]["zip_size"] = zip_size
            TASKS[task_id]["message"] = "Ready for download"
            
        except Exception as e:
            print(f"Export failed: {e}")
            TASKS[task_id]["status"] = "failed"
            TASKS[task_id]["error"] = str(e)

    background_tasks.add_task(run_export_task)

    return {
        "task_id": task_id,
        "status": "processing"
    }

@app.get("/api/exports/download/{task_id}")
def download_export(task_id: str):
    """Download exported dataset as a zip file (pre-built during export)"""
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Export not yet completed")
    
    zip_path = task.get("zip_path")
    if not zip_path or not Path(zip_path).exists():
        # Fallback: build zip now if pre-zip somehow failed
        output_path = task.get("output_path")
        if not output_path or not Path(output_path).exists():
            raise HTTPException(status_code=404, detail="Export files not found on server")
        zip_path = shutil.make_archive(output_path, 'zip', output_path)
    
    export_name = Path(task.get("output_path", "export")).name
    return FileResponse(
        str(zip_path),
        media_type='application/zip',
        filename=export_name + '.zip'
    )

@app.get("/api/annotations/{annotation_id}")
def get_annotation(
    annotation_id: int,
    db: Session = Depends(get_db)
):
    """Get specific annotation version"""
    annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")
    
    return {
        "id": annotation.id,
        "image_id": annotation.image_id,
        "version": annotation.version,
        "data": annotation.data,
        "created_by": annotation.created_by,
        "created_at": annotation.created_at.isoformat(),
        "locked": annotation.locked
    }

@app.post("/api/annotations/{annotation_id}/lock")
def lock_annotation(
    annotation_id: int,
    lock_req: AnnotationLockRequest,
    db: Session = Depends(get_db)
):
    """Lock or unlock annotation version"""
    annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")
    
    annotation.locked = lock_req.locked
    db.commit()
    
    return {
        "id": annotation.id,
        "locked": annotation.locked,
        "version": annotation.version
    }

@app.get("/api/projects/{project_id}/stats")
def get_project_stats(
    project_id: int,
    db: Session = Depends(get_db)
):
    """Get project statistics"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    total_images = len(project.images)
    annotated_images = sum(1 for img in project.images if img.latest_annotation)
    
    # Count annotations by creator
    human_ann = 0
    auto_ann = 0
    for img in project.images:
        for ann in img.annotations:
            if ann.created_by == "human":
                human_ann += len(ann.data)
            else:
                auto_ann += len(ann.data)
    
    return {
        "project_id": project_id,
        "project_name": project.name,
        "annotation_type": project.annotation_type,
        "total_images": total_images,
        "annotated_images": annotated_images,
        "annotation_count": {
            "human": human_ann,
            "auto": auto_ann,
            "total": human_ann + auto_ann
        },
        "class_distribution": {}  # Would implement based on label_schema
    }

# Health check endpoint
@app.get("/api/health")
def health_check():
    """System health check"""
    db_path = WORKSPACE_PATH / "meta.db"
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "workspace": str(WORKSPACE_PATH.absolute()),
        "database": "connected" if db_path.exists() else "missing"
    }

# Root endpoint - serve frontend
@app.get("/")
def root():
    """Serve frontend HTML"""
    return FileResponse("static/index.html")
@app.delete("/api/projects/{project_id}")
def delete_project(
    project_id: int, 
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a project and all its data"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # 1. Delete associated data from DB (Cascading usually handles this, but good to be explicit)
    # Delete annotations logic if not cascaded
    # Delete images logic
    
    # 2. Delete files on disk
    workspace = ProjectWorkspace(WORKSPACE_PATH, project.name)
    try:
        if workspace.project_dir.exists():
            shutil.rmtree(workspace.project_dir)
    except Exception as e:
        print(f"Error deleting files for project {project.name}: {e}")
        # Continue to delete DB record even if file deletion fails
    
    # Save name for logging
    p_name = project.name
    
    # 3. Delete DB record
    db.delete(project)
    db.commit()
    
    # Log activity
    log_activity(db, current_user.id, "delete_project", details={"name": p_name})
    
    return {"message": "Project deleted successfully"}


# ====== AI Auto-Annotation Endpoints ======

class AutoAnnotateRequest(BaseModel):
    use_yolo_world: bool = True
    use_grounding_dino: bool = True
    min_confidence: float = 0.3
    nms_iou_threshold: float = 0.5

@app.post("/api/images/{image_id}/auto-annotate")
async def auto_annotate_image(
    image_id: int,
    request: AutoAnnotateRequest,
    db: Session = Depends(get_db)
):
    """Run AI auto-annotation on a single image"""
    from backend.ai_ensemble import ai_ensemble
    
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    project = image.project
    image_path = WORKSPACE_PATH / image.filepath
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    
    # Get classes from project schema
    classes = project.label_schema.get('classes', [])
    if not classes:
        raise HTTPException(status_code=400, detail="Project has no classes defined")
    
    try:
        # Run ensemble auto-annotation
        result = await ai_ensemble.auto_annotate_ensemble(
            str(image_path),
            classes,
            use_yolo_world=request.use_yolo_world,
            use_grounding_dino=request.use_grounding_dino,
            nms_iou_threshold=request.nms_iou_threshold,
            min_confidence=request.min_confidence,
            custom_model_path=project.label_schema.get('latest_model'),
            project_name=project.name,
            output_type=project.annotation_type
        )
        
        # Save AI predictions to database as "auto" so corrections can be detected later
        if result["annotations"]:
            from sqlalchemy import func
            max_version = db.query(func.max(Annotation.version)).filter(
                Annotation.image_id == image_id
            ).scalar()
            next_version = (max_version or 0) + 1
            
            db_annotation = Annotation(
                image_id=image_id,
                version=next_version,
                data=result["annotations"],
                created_by='auto'
            )
            db.add(db_annotation)
            
            # Save to filesystem
            workspace = ProjectWorkspace(WORKSPACE_PATH, project.name)
            workspace.save_annotation(
                image.filename,
                next_version,
                result["annotations"]
            )
            
            image.verification_status = "unverified"
            db.commit()
            logger.info(f"[Project: {project.name}] 🤖 AI annotation saved for Image {image_id} (Version {next_version})")
        
        return {
            "image_id": image_id,
            "annotations": result["annotations"],
            "count": len(result["annotations"]),
            "models_used": result["models_used"],
            "custom_model_active": result["custom_model_active"]
        }

    except Exception as e:
        logger.error(f"[Project: {project.name}] ❌ Auto-annotation failed for Image {image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Auto-annotation failed: {str(e)}")


@app.post("/api/images/{image_id}/segment")
async def segment_image_object(
    image_id: int,
    request: SegmentationRequest,
    db: Session = Depends(get_db)
):
    """Run interactive SAM segmentation on a specific box"""
    from backend.ai_ensemble import ai_ensemble
    import cv2
    import numpy as np

    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
        
    image_path = WORKSPACE_PATH / image.filepath
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    # Load image for dimension verification and SAM input
    img = cv2.imread(str(image_path))
    if img is None:
        raise HTTPException(status_code=500, detail="Could not read image file")

    # Determine if it's a box or point prompt
    boxes_xyxy = None
    points_np = None
    labels_np = None

    if request.width is not None and request.height is not None:
        # Box Prompt
        x1, y1 = request.x, request.y
        x2, y2 = x1 + request.width, y1 + request.height
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        boxes_xyxy = np.array([[x1, y1, x2, y2]])
    elif request.points:
        # Multi-point prompt
        points_np = np.array([[p['x'], p['y']] for p in request.points])
        labels_np = np.ones(len(points_np))
    else:
        # Single click point prompt
        points_np = np.array([[request.x, request.y]])
        labels_np = np.ones(1)
    
    # Run SAM
    polygons = ai_ensemble.run_sam(
        img, 
        boxes_xyxy=boxes_xyxy, 
        points=points_np, 
        labels=labels_np
    )
    
    if not polygons or not polygons[0]:
        raise HTTPException(status_code=422, detail="SAM failed to generate a segment for this location")
        
    return {
        "type": "polygon",
        "points": polygons[0]
    }


async def internal_run_batch_annotation(
    project_id: int,
    task_id: str,
    request: AutoAnnotateRequest,
    db: Session
):
    """Internal helper to run batch annotation, usable by API and background tasks"""
    from backend.ai_ensemble import ai_ensemble
    
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        logger.error(f"Project {project_id} not found for batch task {task_id}")
        return

    # Get images: either no annotations, or only AI annotations (unverified)
    images = []
    for img in project.images:
        latest = img.latest_annotation
        if not latest or latest.created_by == "auto":
            images.append(img)
    
    if not images:
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["message"] = "No images need annotation"
        return

    TASKS[task_id]["total"] = len(images)
    classes = project.label_schema.get('classes', [])
    
    import asyncio
    # Note: We assume this is called within an existing event loop or handled by BackgroundTasks
    
    for i, img in enumerate(images):
        try:
            image_path = WORKSPACE_PATH / img.filepath
            
            # Run annotation
            ensemble_result = await ai_ensemble.auto_annotate_ensemble(
                str(image_path),
                classes,
                use_yolo_world=request.use_yolo_world,
                use_grounding_dino=request.use_grounding_dino,
                nms_iou_threshold=request.nms_iou_threshold,
                min_confidence=request.min_confidence,
                custom_model_path=project.label_schema.get('latest_model'),
                output_type=project.annotation_type
            )
            
            if i == 0 or i % 10 == 0:
                model_type = "Custom Model" if ensemble_result["custom_model_active"] else "Base Models"
                TASKS[task_id]["message"] = f"Annotating with {model_type} ({i+1}/{len(images)})"
            
            annotations = ensemble_result["annotations"]
            
            # Save annotations as 'auto'
            if annotations:
                workspace = ProjectWorkspace(WORKSPACE_PATH, project.name)
                version = workspace.get_next_annotation_version(img.filename)
                
                db_ann = Annotation(
                    image_id=img.id,
                    version=version,
                    data=annotations,
                    created_by="auto"
                )
                db.add(db_ann)
                img.verification_status = "unverified"
                db.commit()
            
            TASKS[task_id]["completed_images"].append({
                "id": img.id,
                "filename": img.filename,
                "count": len(annotations)
            })
            
        except Exception as e:
            logger.error(f"Error annotating {img.filename} in batch {task_id}: {e}")
        
        TASKS[task_id]["current"] = i + 1
        TASKS[task_id]["progress"] = int(((i + 1) / len(images)) * 100)
    
    TASKS[task_id]["status"] = "completed"
    TASKS[task_id]["message"] = f"Batch complete! Annotated {len(images)} images."

@app.post("/api/projects/{project_id}/auto-annotate-batch")
async def auto_annotate_batch(
    project_id: int,
    request: AutoAnnotateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start batch auto-annotation for all pending/unverified images in a project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Quick count for immediate feedback
    count = 0
    for img in project.images:
        latest = img.latest_annotation
        if not latest or latest.created_by == "auto":
            count += 1
            
    if count == 0:
        return {"message": "No unverified images to annotate", "count": 0}
    
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "project_id": project_id,
        "status": "processing",
        "progress": 0,
        "total": count,
        "current": 0,
        "completed_images": [],
        "message": "Initializing batch task..."
    }
    
    async def run_batch_wrapper():
        db_local = db_models.SessionLocal()
        try:
            await internal_run_batch_annotation(project_id, task_id, request, db_local)
        except Exception as e:
            TASKS[task_id]["status"] = "failed"
            TASKS[task_id]["error"] = str(e)
            logger.error(f"Batch task {task_id} failed: {e}")
        finally:
            db_local.close()

    background_tasks.add_task(run_batch_wrapper)
    
    return {
        "task_id": task_id,
        "message": f"Started batch annotation for {count} images",
        "total": count
    }


@app.get("/api/ai/status")
def get_ai_status():
    """Get status of AI models"""
    try:
        from backend.ai_ensemble import ai_ensemble
        
        return {
            "initialized": ai_ensemble.initialized,
            "device": ai_ensemble.device,
            "available_models": ai_ensemble.get_available_models(),
            "ready": ai_ensemble.is_ready()
        }
    except Exception as e:
        return {
            "initialized": False,
            "error": str(e)
        }


@app.post("/api/ai/initialize")
def initialize_ai():
    """Manually initialize AI models"""
    try:
        from backend.ai_ensemble import ai_ensemble
        ai_ensemble.initialize_models(force=True)
        
        return {
            "status": "success",
            "models": ai_ensemble.get_available_models()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize AI: {str(e)}")


# ====== Training & Augmentation Endpoints ======

class TrainingRequest(BaseModel):
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    augment_multiplier: int = 3  # How many augmented copies per verified image

@app.post("/api/projects/{project_id}/train")
async def train_project_model(
    project_id: int,
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start YOLO training with data augmentation"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Check if a training job is already running for this project
    active_tasks = [t_id for t_id, t in TASKS.items() if t.get("project_id") == project_id and t.get("status") not in ["completed", "failed"]]
    if active_tasks:
        raise HTTPException(status_code=400, detail="A training job is already active for this project")

    # Check if we have verified images (human annotations)
    verified_images = []
    for img in project.images:
        if img.latest_annotation and img.latest_annotation.created_by == "human":
            verified_images.append(img)
    
    if len(verified_images) < 5:
        raise HTTPException(status_code=400, detail=f"Need at least 5 verified images to start training (found {len(verified_images)})")

    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "project_id": project_id,
        "status": "augmenting",
        "progress": 0,
        "message": "Generating augmented images..."
    }
    def run_training_flow():
        from backend.augmentation import augmentor
        from ultralytics import YOLO
        import yaml
        import torch
        import os
        
        # 0. Lower process priority and limit threads to prevent system freeze
        try:
            os.nice(10) # Lower priority (niceness)
            torch.set_num_threads(4) # Limit CPU threads to 4
        except:
            pass
        
        db_local = db_models.SessionLocal()
        try:
            # 0.5 Re-fetch project and verified images in local session to avoid detached session issues
            project_local = db_local.query(Project).filter(Project.id == project.id).first()
            if not project_local:
                logger.error(f"[Project: {project.id}] ❌ Project not found in background task session")
                return
            
            logger.info(f"[Project: {project_local.name}] ⚡ CPU Optimization: Throttling training process (nice=10, threads=4)")
            
            # Re-fetch verified images to ensure they are attached to db_local
            verified_images_local = []
            for img in project_local.images:
                if img.latest_annotation and img.latest_annotation.created_by == "human":
                    verified_images_local.append(img)
            
            if len(verified_images_local) < 5:
                logger.error(f"[Project: {project_local.name}] ❌ Not enough verified images in background task ({len(verified_images_local)})")
                return

            # 1. Setup training workspace
            training_dir = WORKSPACE_PATH / project_local.name / "training"
            if training_dir.exists():
                shutil.rmtree(training_dir, ignore_errors=True)
            training_dir.mkdir(parents=True, exist_ok=True)
            
            # Sub-dirs
            images_dir = training_dir / "images"
            labels_dir = training_dir / "labels"
            images_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)
            
            # 2. Augment images
            processed_count = 0
            total_images = len(verified_images_local)
            is_seg = project_local.annotation_type == "polygon"
            
            for img in verified_images_local:
                source_path = WORKSPACE_PATH / img.filepath
                if not source_path.exists():
                    continue
                
                # Copy original
                shutil.copy2(source_path, images_dir / img.filename)
                
                # Save Label Helper
                def save_yolo_label(target_path, anns):
                    with open(target_path, 'w') as f:
                        for ann in anns:
                            if is_seg and 'points' in ann:
                                # YOLO Segment format: class_id x1 y1 x2 y2 ...
                                pts = " ".join([f"{p['x']:.6f} {p['y']:.6f}" for p in ann['points']])
                                f.write(f"{ann['class_id']} {pts}\n")
                            else:
                                # YOLO Detection format: class_id xc yc w h
                                xc = ann['x'] + ann['width'] / 2
                                yc = ann['y'] + ann['height'] / 2
                                f.write(f"{ann['class_id']} {xc:.6f} {yc:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")

                # Save original label
                save_yolo_label(labels_dir / f"{Path(img.filename).stem}.txt", img.latest_annotation.data)
                
                # Generate augmented
                if request.augment_multiplier > 0:
                    aug_results = augmentor.generate_augmented_batch(
                        str(source_path),
                        img.latest_annotation.data,
                        images_dir,
                        Path(img.filename).stem,
                        multiplier=request.augment_multiplier
                    )
                    
                    for aug_path, aug_anns in aug_results:
                        save_yolo_label(labels_dir / f"{Path(aug_path).stem}.txt", aug_anns)
                
                processed_count += 1
                TASKS[task_id]["progress"] = int((processed_count / total_images) * 40)
                if processed_count % 5 == 0 or processed_count == total_images:
                    logger.info(f"[Project: {project_local.name}] 🛠️ Data Prep: Processed {processed_count}/{total_images} images with augmentations.")
            
            # 3. Create data.yaml
            classes = project_local.label_schema.get('classes', [])
            class_map = {c['id']: c['name'] for c in classes}
            
            data_yaml = {
                'path': str(training_dir.absolute()),
                'train': 'images',
                'val': 'images', 
                'names': class_map
            }
            
            with open(training_dir / 'data.yaml', 'w') as f:
                yaml.dump(data_yaml, f)
            
            # 4. Start YOLO training
            TASKS[task_id]["status"] = "training"
            TASKS[task_id]["message"] = f"Initializing YOLO {'Segment' if is_seg else 'Detect'} training..."
            
            # Base model - Resume from latest if available
            is_seg = project_local.annotation_type == "polygon"
            latest_model_path = project_local.label_schema.get('latest_model')
            
            if latest_model_path and Path(latest_model_path).exists():
                base_model = latest_model_path
                logger.info(f"[Project: {project_local.name}] 🔄 Resuming training from latest custom model: {base_model}")
            else:
                base_model = 'yolov8n-seg.pt' if is_seg else 'yolov8n.pt'
                logger.info(f"[Project: {project_local.name}] 🆕 Starting new training from base model: {base_model}")
            
            logger.info(f"[Project: {project_local.name}] 🚀 Training Specs: Epochs={request.epochs}, Batch={request.batch}, Imgsz={request.imgsz}, Task={'segment' if is_seg else 'detect'}")
            
            model = YOLO(base_model) 
            
            # Add callback to track epochs
            def on_train_epoch_end(trainer):
                epoch = trainer.epoch + 1
                total_epochs = trainer.epochs
                TASKS[task_id]["epoch"] = epoch
                TASKS[task_id]["total_epochs"] = total_epochs
                # Progress spans 40% to 100% during training (first 40% was data prep)
                TASKS[task_id]["progress"] = int(40 + (epoch / total_epochs) * 60)
                TASKS[task_id]["message"] = f"Epoch {epoch}/{total_epochs}"
            
            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            results = model.train(
                data=str(training_dir / 'data.yaml'),
                epochs=request.epochs,
                imgsz=request.imgsz,
                batch=request.batch,
                device='cpu',
                workers=2,         # Reduce data loader workers
                exist_ok=True,     # Reuse directory if it exists
                project=str(training_dir / 'runs'),
                name='exp',
                task='segment' if is_seg else 'detect'
            )
            
            # 5. Save best model and refresh AI cache
            best_model = Path(results.save_dir) / 'weights' / 'best.pt'
            if best_model.exists():
                project_model_dir = WORKSPACE_PATH / project_local.name / "models"
                project_model_dir.mkdir(exist_ok=True)
                final_model_path = project_model_dir / "latest_finetuned.pt"
                shutil.copy2(best_model, final_model_path)
                
                # Update schema
                from sqlalchemy.orm.attributes import flag_modified
                schema = project_local.label_schema.copy()
                schema["latest_model"] = str(final_model_path.absolute())
                schema["last_trained_image_count"] = len(verified_images_local)
                schema["last_trained_correction_count"] = schema.get('correction_count', 0)
                project_local.label_schema = schema
                flag_modified(project_local, "label_schema")
                db_local.commit()
                logger.info(f"[Project: {project_local.name}] ✅ Training complete. Model updated and schema saved.")
                
                # ♻️ Refresh AI Engine Cache
                from backend.ai_ensemble import ai_ensemble
                ai_ensemble.clear_model_cache(str(final_model_path))

                # 🚀 Auto Re-annotate unverified images with the brand new model
                logger.info(f"[Project: {project_local.name}] 🔄 Triggering automatic re-annotation with new model...")
                reann_task_id = f"reann_{task_id}"
                TASKS[reann_task_id] = {
                    "project_id": project_local.id,
                    "status": "processing",
                    "progress": 0,
                    "total": 0,
                    "current": 0,
                    "completed_images": [],
                    "message": "AI self-improvement: Re-annotating with new model..."
                }
                
                # We need to run the async re-annotation in the same way batch does
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Use default settings for re-annotation: focus on the NEW custom model
                    # (YOLO-World and Grounding DINO can still be used if enabled in common settings, 
                    # but here we simplify to the new best weights)
                    reann_request = AutoAnnotateRequest(
                        use_yolo_world=False,
                        use_grounding_dino=False,
                        min_confidence=0.3
                    )
                    loop.run_until_complete(
                        internal_run_batch_annotation(project_local.id, reann_task_id, reann_request, db_local)
                    )
                finally:
                    loop.close()

            TASKS[task_id]["status"] = "completed"
            TASKS[task_id]["progress"] = 100
            TASKS[task_id]["message"] = "Training and re-annotation finished!"
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            TASKS[task_id]["status"] = "failed"
            TASKS[task_id]["error"] = str(e)
        finally:
            db_local.close()
            
    background_tasks.add_task(run_training_flow)
    
    return {
        "task_id": task_id,
        "message": "Training started in background",
        "verified_images": len(verified_images)
    }

# Auto-Training Endpoints
@app.get("/api/projects/{project_id}/check-auto-train")
async def check_auto_train_eligibility(
    project_id: int,
    db: Session = Depends(get_db)
):
    """Check if project is eligible for automatic training"""
    from backend.training_orchestrator import TrainingOrchestrator
    
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    result = TrainingOrchestrator.check_auto_train_eligibility(project, db)
    return result

@app.get("/api/projects/{project_id}/correction-stats")
async def get_correction_stats(
    project_id: int,
    db: Session = Depends(get_db)
):
    """Get correction statistics for fine-tuning eligibility"""
    from backend.training_orchestrator import TrainingOrchestrator
    
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    result = TrainingOrchestrator.check_fine_tune_eligibility(project, db)
    return result

# ==================== ANALYTICS ====================

@app.get("/api/analytics/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get system-wide stats for dashboard"""
    total_projects = db.query(Project).count()
    total_images = db.query(Image).count()
    total_annotations = db.query(Annotation).count()
    total_users = db.query(db_models.User).count()
    
    counts_by_type = {}
    for t in AnnotationType:
        counts_by_type[t.value] = db.query(Project).filter(Project.annotation_type == t).count()
        
    return {
        "global": {
            "projects": total_projects,
            "images": total_images,
            "annotations": total_annotations,
            "users": total_users
        },
        "by_type": counts_by_type
    }

@app.get("/api/projects/{project_id}/analytics")
async def get_project_analytics(project_id: int, db: Session = Depends(get_db)):
    """Get detailed analytics for a specific project"""
    from sqlalchemy import func, distinct
    
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    total_images = db.query(Image).filter(Image.project_id == project_id).count()
    
    # Count unique images with human annotations
    human_annotated = db.query(func.count(distinct(Annotation.image_id))).join(
        Image, Annotation.image_id == Image.id
    ).filter(
        Image.project_id == project_id,
        Annotation.created_by == "human"
    ).scalar() or 0
    
    # Count unique images with auto annotations
    ai_annotated = db.query(func.count(distinct(Annotation.image_id))).join(
        Image, Annotation.image_id == Image.id
    ).filter(
        Image.project_id == project_id,
        Annotation.created_by == "auto"
    ).scalar() or 0
    
    # Count corrections (human annotation saved after auto on same image)
    correction_count = project.label_schema.get('correction_count', 0)
    
    # AI accuracy estimate
    ai_correct = max(0, ai_annotated - correction_count) if ai_annotated > 0 else 0
    ai_accuracy = round((ai_correct / ai_annotated * 100), 1) if ai_annotated > 0 else 0
    
    # Time estimates (30s per manual annotation, 2s per auto)
    manual_time_sec = human_annotated * 30
    auto_time_sec = ai_annotated * 2
    time_saved_sec = max(0, (ai_annotated * 30) - auto_time_sec) if ai_annotated > 0 else 0
    
    # Verification status counts
    verified_count = db.query(Image).filter(
        Image.project_id == project_id,
        Image.verification_status == "verified"
    ).count()
    unverified_count = db.query(Image).filter(
        Image.project_id == project_id,
        Image.verification_status == "unverified"
    ).count()
    needs_edit_count = db.query(Image).filter(
        Image.project_id == project_id,
        Image.verification_status == "needs_edit"
    ).count()
    
    return {
        "project_name": project.name,
        "total_images": total_images,
        "human_annotations": human_annotated,
        "ai_annotations": ai_annotated,
        "corrections": correction_count,
        "ai_correct_detections": ai_correct,
        "ai_accuracy_percent": ai_accuracy,
        "time": {
            "manual_annotation_sec": manual_time_sec,
            "auto_annotation_sec": auto_time_sec,
            "time_saved_sec": time_saved_sec,
            "manual_annotation_display": f"{manual_time_sec // 60}m {manual_time_sec % 60}s",
            "auto_annotation_display": f"{auto_time_sec // 60}m {auto_time_sec % 60}s",
            "time_saved_display": f"{time_saved_sec // 60}m {time_saved_sec % 60}s"
        },
        "verification": {
            "verified": verified_count,
            "unverified": unverified_count,
            "needs_edit": needs_edit_count,
            "progress_percent": round((verified_count / total_images * 100), 1) if total_images > 0 else 0
        }
    }

# ==================== IMAGE VERIFICATION ====================

@app.patch("/api/images/{image_id}/verify")
async def verify_image(image_id: int, db: Session = Depends(get_db)):
    """Mark an image as verified"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    image.verification_status = "verified"
    db.commit()
    return {"status": "verified", "image_id": image_id}

@app.patch("/api/images/{image_id}/needs-edit")
async def mark_needs_edit(image_id: int, db: Session = Depends(get_db)):
    """Mark a verified image as needing edits"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    image.verification_status = "needs_edit"
    db.commit()
    return {"status": "needs_edit", "image_id": image_id}

@app.patch("/api/images/{image_id}/unverify")
async def unverify_image(image_id: int, db: Session = Depends(get_db)):
    """Mark an image as unverified (after model re-annotation)"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    image.verification_status = "unverified"
    db.commit()
    return {"status": "unverified", "image_id": image_id}

# ==================== ACTIVITY LOGS ====================

@app.get("/api/logs/activity")
async def get_activity_logs(limit: int = 100, db: Session = Depends(get_db)):
    """Get user activity logs (info level)"""
    logs = db.query(ActivityLog).filter(
        ActivityLog.level == "info"
    ).order_by(ActivityLog.created_at.desc()).limit(limit).all()
    results = []
    for log in logs:
        results.append({
            "id": log.id,
            "username": log.user.username if log.user else "System",
            "action": log.action,
            "project_name": log.project.name if log.project else None,
            "details": log.details,
            "created_at": log.created_at.isoformat()
        })
    return results

@app.get("/api/logs/errors")
async def get_error_logs(limit: int = 100, db: Session = Depends(get_db)):
    """Get system error logs for debugging"""
    logs = db.query(ActivityLog).filter(
        ActivityLog.level == "error"
    ).order_by(ActivityLog.created_at.desc()).limit(limit).all()
    results = []
    for log in logs:
        results.append({
            "id": log.id,
            "username": log.user.username if log.user else "System",
            "action": log.action,
            "project_name": log.project.name if log.project else None,
            "details": log.details,
            "traceback": log.traceback,
            "created_at": log.created_at.isoformat()
        })
    return results

# Keep old endpoint for backwards compat
@app.get("/api/analytics/logs")
async def get_logs_compat(limit: int = 100, db: Session = Depends(get_db)):
    """Get activity logs (backwards compatible)"""
    logs = db.query(ActivityLog).order_by(ActivityLog.created_at.desc()).limit(limit).all()
    results = []
    for log in logs:
        results.append({
            "id": log.id,
            "username": log.user.username if log.user else "System",
            "action": log.action,
            "project_name": log.project.name if log.project else None,
            "details": log.details,
            "level": log.level or "info",
            "traceback": log.traceback,
            "created_at": log.created_at.isoformat()
        })
    return results
