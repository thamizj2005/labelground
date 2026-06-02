# File upload endpoints - Add these to main.py after installing python-multipart
# Run: conda install -y python-multipart -c conda-forge

@app.post("/api/projects/{project_id}/upload-images")
async def upload_images(
    project_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Upload images directly from browser"""
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
    db: Session = Depends(get_db)
):
    """Upload video directly from browser"""
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
    
    def extract_frames_task():
        try:
            frame_paths = VideoProcessor.extract_frames(
                video_dest,
                workspace.images_dir,
                fps,
                progress_callback=None
            )
            
            # Register frames in database
            db_local = db_models.SessionLocal()
            try:
                for frame_path in frame_paths:
                    width, height = ImageImporter.get_image_dimensions(frame_path)
                    
                    db_image = Image(
                        project_id=project_id,
                        filename=frame_path.name,
                        filepath=str(frame_path.relative_to(WORKSPACE_PATH)),
                        width=width,
                        height=height,
                        status=ImageStatus.PROCESSED
                    )
                    db_local.add(db_image)
                
                db_local.commit()
            finally:
                db_local.close()
                
        except Exception as e:
            print(f"Video extraction failed: {e}")
    
    background_tasks.add_task(extract_frames_task)
    
    return {
        "task_id": task_id,
        "video_path": str(video_dest),
        "estimated_frames": estimated_frames,
        "target_fps": fps,
        "original_fps": video_fps,
        "status": "processing"
    }
