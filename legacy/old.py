import sys
import os
import cv2
import torch
import numpy as np
import json
import supervision as sv
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTextEdit, QLineEdit, QSlider, QCheckBox, QComboBox,
                             QSpinBox, QGroupBox, QProgressBar, QDoubleSpinBox,
                             QRadioButton, QButtonGroup, QDialog, QGraphicsView,
                             QGraphicsScene, QGraphicsRectItem, QGraphicsPathItem, QGraphicsTextItem,
                             QListWidget, QListWidgetItem, QSplitter, QFrame,
                             QGridLayout, QMessageBox, QToolBar, QMenu, QMenuBar,
                             QColorDialog, QStyleFactory, QStyle, QScrollArea)
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QPen, QBrush, QColor, 
                        QFont, QKeySequence, QAction, QCursor, QIcon, QPainterPath, QPolygonF)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF, QTimer
from PIL import Image
import random
import shutil

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  Ultralytics not installed. YOLOv8 detection disabled.")

from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import SamPredictor, sam_model_registry

# ==============================================================================
# CONFIG
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_DIR = "./weights"
GD_CONFIG = os.path.join(WEIGHTS_DIR, "GroundingDINO_SwinT_OGC.py")
GD_CHECKPOINT = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
SAM_CHECKPOINT = os.path.join(WEIGHTS_DIR, "sam_vit_b_01ec64.pth")
YOLO_CHECKPOINT = "./yolov8s-seg.pt"

# YOLO COCO class names (80 classes)
YOLO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class AnnotationItem(QGraphicsPathItem):
    """Custom graphics item for annotations (Box or Polygon)"""
    def __init__(self, rect, mask=None, label="", class_id=0, color=None):
        super().__init__()
        self.rect_data = rect # Store original rect for reference
        self.mask_data = mask
        self.label = label
        self.class_id = class_id
        self.color = color if color else QColor(random.randint(50, 200), 
                                               random.randint(50, 200), 
                                               random.randint(50, 200))
        
        # Create Path
        path = QPainterPath()
        if mask is not None and len(mask) > 0:
            # Polygon mask
            path.moveTo(mask[0][0], mask[0][1])
            for point in mask[1:]:
                path.lineTo(point[0], point[1])
            path.closeSubpath()
        else:
            # Rectangle
            path.addRect(rect)
        
        self.setPath(path)
        
        # Set appearance
        pen = QPen(self.color, 3)
        pen.setStyle(Qt.PenStyle.DashLine)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(self.color.red(), self.color.green(), self.color.blue(), 60)))
        
        # Create label text
        self.text_item = QGraphicsTextItem(label, self)
        self.text_item.setFont(QFont("Arial", 8))
        self.text_item.setDefaultTextColor(QColor(255, 255, 255))
        self.update_text_position()
        
        # Make movable and selectable
        self.setFlag(QGraphicsPathItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsPathItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsPathItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        
    def update_text_position(self):
        """Update label position at top-left corner"""
        rect = self.path().boundingRect()
        self.text_item.setPos(rect.x(), rect.y() - 15)
        
    def itemChange(self, change, value):
        if change == QGraphicsPathItem.GraphicsItemChange.ItemPositionChange:
            # Only update text if relative position works (it's a child, so it moves with parent)
            # But if we wanted to enforce screen alignment we'd do it here.
            pass
        return super().itemChange(change, value)

class ManualAnnotationDialog(QDialog):
    """Dialog for manual annotation editing"""
    def __init__(self, image_path, initial_boxes=[], initial_labels=[], 
                 initial_masks=[], class_names=None, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.class_names = class_names or ["object"]
        self.initial_boxes = initial_boxes
        self.initial_labels = initial_labels
        self.initial_masks = initial_masks
        self.current_class = 0
        self.drawing = False
        self.start_point = None
        self.current_item = None
        self.annotation_items = []
        
        self.setWindowTitle("Manual Annotation Editor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Load image
        self.cv_image = cv2.imread(image_path)
        if self.cv_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.init_ui()
        self.load_initial_annotations(self.initial_boxes, self.initial_labels, self.initial_masks)
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Toolbar
        toolbar = QToolBar()
        
        # Class selection
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.class_names)
        self.class_combo.currentIndexChanged.connect(self.on_class_changed)
        toolbar.addWidget(QLabel("Class:"))
        toolbar.addWidget(self.class_combo)
        
        # Drawing mode buttons
        self.draw_btn = QPushButton("✏️ Draw Box")
        self.draw_btn.setCheckable(True)
        self.draw_btn.toggled.connect(self.on_draw_mode_toggled)
        toolbar.addWidget(self.draw_btn)
        
        self.select_btn = QPushButton("👆 Select")
        self.select_btn.setCheckable(True)
        self.select_btn.toggled.connect(self.on_select_mode_toggled)
        toolbar.addWidget(self.select_btn)
        
        # Delete button
        delete_btn = QPushButton("🗑️ Delete Selected")
        delete_btn.clicked.connect(self.delete_selected)
        toolbar.addWidget(delete_btn)
        
        # Color picker
        color_btn = QPushButton("🎨 Color")
        color_btn.clicked.connect(self.pick_color)
        toolbar.addWidget(color_btn)
        
        toolbar.addSeparator()
        
        # Zoom buttons
        zoom_in_btn = QPushButton("🔍 +")
        zoom_in_btn.clicked.connect(self.zoom_in)
        toolbar.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("🔍 -")
        zoom_out_btn.clicked.connect(self.zoom_out)
        toolbar.addWidget(zoom_out_btn)
        
        layout.addWidget(toolbar)
        
        # Split view
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Graphics view for annotation
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setMouseTracking(True)
        self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        
        # Load image into scene
        self.display_image()
        
        # Install event filter for mouse events
        self.view.viewport().installEventFilter(self)
        
        splitter.addWidget(self.view)
        
        # Side panel for annotation list
        side_panel = QWidget()
        side_layout = QVBoxLayout()
        
        self.annotation_list = QListWidget()
        self.annotation_list.itemSelectionChanged.connect(self.on_list_selection_changed)
        side_layout.addWidget(QLabel("Annotations:"))
        side_layout.addWidget(self.annotation_list)
        
        side_panel.setLayout(side_layout)
        splitter.addWidget(side_panel)
        
        layout.addWidget(splitter)
        
        # Button box
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("💾 Save & Close")
        save_btn.clicked.connect(self.accept)
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def eventFilter(self, obj, event):
        """Handle mouse events for drawing"""
        if obj is self.view.viewport():
            if self.draw_btn.isChecked():
                if event.type() == event.Type.MouseButtonPress:
                    self.handle_mouse_press(event)
                    return True
                elif event.type() == event.Type.MouseMove:
                    self.handle_mouse_move(event)
                    return True
                elif event.type() == event.Type.MouseButtonRelease:
                    self.handle_mouse_release(event)
                    return True
        return super().eventFilter(obj, event)
        
    def display_image(self):
        """Display the image in the graphics view"""
        height, width = self.cv_image.shape[:2]
        bytes_per_line = 3 * width
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qimage)
        self.scene.addPixmap(pixmap)
        
        # Set scene rect to image size
        self.scene.setSceneRect(0, 0, width, height)
        
    def load_initial_annotations(self, boxes, labels, masks):
        """Load initial annotations from AI detection"""
        for i, (box, label) in enumerate(zip(boxes, labels)):
            mask = masks[i] if i < len(masks) else None
            
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            # Find class id
            try:
                class_id = self.class_names.index(label)
            except ValueError:
                class_id = 0
                
            item = AnnotationItem(QRectF(x1, y1, w, h), mask, label, class_id)
            self.scene.addItem(item)
            self.annotation_items.append(item)
            
            # Add to list
            list_item = QListWidgetItem(f"{label}: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
            self.annotation_list.addItem(list_item)
    
    def on_draw_mode_toggled(self, checked):
        """Handle draw mode toggle"""
        if checked:
            self.select_btn.setChecked(False)
            self.view.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.view.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
    
    def on_select_mode_toggled(self, checked):
        """Handle select mode toggle"""
        if checked:
            self.draw_btn.setChecked(False)
            self.view.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
    
    def on_class_changed(self, index):
        """Handle class change"""
        self.current_class = index
    
    def handle_mouse_press(self, event):
        """Handle mouse press for drawing"""
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.view.mapToScene(event.pos())
            self.start_point = scene_pos
            self.drawing = True
            # Passing None as mask for new manual rectangle
            self.current_item = AnnotationItem(
                QRectF(scene_pos, scene_pos), # Pass Rect, not QPainterPath
                None, # No mask data initially
                self.class_names[self.current_class],
                self.current_class
            )
            self.scene.addItem(self.current_item)
    
    def handle_mouse_move(self, event):
        """Handle mouse move for drawing"""
        if self.drawing and self.current_item:
            scene_pos = self.view.mapToScene(event.pos())
            rect = QRectF(self.start_point, scene_pos).normalized()
            # Update path to be a rectangle
            path = QPainterPath()
            path.addRect(rect)
            self.current_item.setPath(path)
    
    def handle_mouse_release(self, event):
        """Handle mouse release for drawing"""
        if event.button() == Qt.MouseButton.LeftButton and self.drawing and self.current_item:
            self.drawing = False
            
            # Check if rectangle has valid size
            # Use boundingRect() instead of rect()
            rect = self.current_item.path().boundingRect()
            if rect.width() > 10 and rect.height() > 10:  # Minimum size check
                # Add to list
                label = self.class_names[self.current_class]
                list_item = QListWidgetItem(
                    f"{label}: [{rect.x():.0f}, {rect.y():.0f}, {rect.right():.0f}, {rect.bottom():.0f}]"
                )
                self.annotation_list.addItem(list_item)
                self.annotation_items.append(self.current_item)
                
                # Update text position
                self.current_item.update_text_position()
                self.current_item = None
            else:
                # Remove if too small
                self.scene.removeItem(self.current_item)
                self.current_item = None
    
    def delete_selected(self):
        """Delete selected annotations"""
        selected_items = self.scene.selectedItems()
        for item in selected_items:
            if isinstance(item, AnnotationItem):
                # Remove from scene
                self.scene.removeItem(item)
                # Remove from list
                for i in range(self.annotation_list.count()):
                    list_item = self.annotation_list.item(i)
                    if item.label in list_item.text():
                        self.annotation_list.takeItem(i)
                        break
                self.annotation_items.remove(item)
    
    def pick_color(self):
        """Pick color for selected annotation"""
        selected = self.scene.selectedItems()
        if selected and isinstance(selected[0], AnnotationItem):
            color = QColorDialog.getColor()
            if color.isValid():
                selected[0].color = color
                pen = QPen(color, 3)
                pen.setStyle(Qt.PenStyle.DashLine)
                selected[0].setPen(pen)
    
    def zoom_in(self):
        """Zoom in"""
        self.view.scale(1.2, 1.2)
    
    def zoom_out(self):
        """Zoom out"""
        self.view.scale(0.8, 0.8)
    
    def on_list_selection_changed(self):
        """Handle list selection changed"""
        selected_items = self.annotation_list.selectedItems()
        if selected_items:
            # Clear current selection in scene
            for item in self.scene.selectedItems():
                item.setSelected(False)
            
            # Select corresponding graphic item
            text = selected_items[0].text()
            for item in self.annotation_items:
                if item.label in text:
                    item.setSelected(True)
                    self.view.centerOn(item)
                    break
    
    def get_annotations(self):
        """Get all annotations as lists"""
        boxes = []
        labels = []
        masks = [] # Collect masks
        confidences = [1.0] * len(self.annotation_items)  # Manual annotations have confidence 1.0
        
        for item in self.annotation_items:
            rect = item.path().boundingRect() # Use boundingRect() for QGraphicsPathItem
            boxes.append([rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height()])
            labels.append(item.label)
            masks.append(item.mask_data) # Collect mask data (might be None for pure boxes)
        
        return boxes, labels, confidences, masks

class BatchReviewDialog(QDialog):
    """Dialog for reviewing batch annotations"""
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results  # List of (image_path, boxes, labels, confidence, annotated_img)
        self.current_index = 0
        self.review_status = {}  # image_path -> 'accept', 'reject', or 'manual'
        
        self.setWindowTitle("Batch Annotation Review")
        self.setGeometry(100, 100, 1400, 800)
        
        self.init_ui()
        self.load_current_image()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Progress indicator
        self.progress_label = QLabel()
        layout.addWidget(self.progress_label)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #444; background-color: #222;")
        layout.addWidget(self.image_label, stretch=3)
        
        # Annotation info
        info_layout = QHBoxLayout()
        
        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-size: 14px; padding: 10px;")
        info_layout.addWidget(self.info_label)
        
        # Stats label
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("font-size: 14px; color: #888; padding: 10px;")
        info_layout.addWidget(self.stats_label, stretch=1)
        
        layout.addLayout(info_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("⏮ Previous")
        self.prev_btn.clicked.connect(self.previous_image)
        button_layout.addWidget(self.prev_btn)
        
        self.accept_btn = QPushButton("✅ Accept & Next")
        self.accept_btn.clicked.connect(self.accept_annotation)
        self.accept_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        button_layout.addWidget(self.accept_btn)
        
        self.reject_btn = QPushButton("❌ Reject")
        self.reject_btn.clicked.connect(self.reject_annotation)
        self.reject_btn.setStyleSheet("background-color: #dc3545; color: white;")
        button_layout.addWidget(self.reject_btn)
        
        self.manual_btn = QPushButton("✏️ Edit Manually")
        self.manual_btn.clicked.connect(self.edit_manually)
        self.manual_btn.setStyleSheet("background-color: #17a2b8; color: white;")
        button_layout.addWidget(self.manual_btn)
        
        self.next_btn = QPushButton("Next ⏭")
        self.next_btn.clicked.connect(self.next_image)
        button_layout.addWidget(self.next_btn)
        
        layout.addLayout(button_layout)
        
        # Navigation shortcuts info
        shortcut_label = QLabel("Shortcuts: ← Previous, → Next, A Accept, R Reject, E Edit, S Skip")
        shortcut_label.setStyleSheet("color: #666; font-size: 12px; padding: 5px;")
        layout.addWidget(shortcut_label)
        
        # Final buttons
        final_layout = QHBoxLayout()
        
        self.save_all_btn = QPushButton("💾 Save All Accepted")
        self.save_all_btn.clicked.connect(self.save_all_accepted)
        final_layout.addWidget(self.save_all_btn)
        
        self.finish_btn = QPushButton("🏁 Finish Review")
        self.finish_btn.clicked.connect(self.accept)
        final_layout.addWidget(self.finish_btn)
        
        layout.addLayout(final_layout)
        
        self.setLayout(layout)
        
        # Enable keyboard shortcuts
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key.Key_Left:
            self.previous_image()
        elif event.key() == Qt.Key.Key_A:
            self.accept_annotation()
        elif event.key() == Qt.Key.Key_R:
            self.reject_annotation()
        elif event.key() == Qt.Key.Key_E:
            self.edit_manually()
        elif event.key() == Qt.Key.Key_S:
            self.skip_image()
        else:
            super().keyPressEvent(event)
    
    def load_current_image(self):
        """Load current image with annotations"""
        if 0 <= self.current_index < len(self.results):
            # Safe unpacking
            result = self.results[self.current_index]
            image_path = result[0]
            boxes = result[1]
            labels = result[2]
            confidence = result[3]
            annotated_img = result[4]
            masks = result[5] if len(result) > 5 else []
            
            # Update progress
            self.progress_label.setText(
                f"Image {self.current_index + 1} of {len(self.results)}: {Path(image_path).name}"
            )
            
            # Display image
            if annotated_img is not None:
                self.display_image(annotated_img)
            else:
                # Show original if no annotations
                cv_img = cv2.imread(image_path)
                if cv_img is not None:
                    self.display_image(cv_img)
                else:
                    self.image_label.setText("Failed to load image")
            
            # Update info - FIXED: Handle empty confidence list
            avg_confidence = 0.0
            if confidence and len(confidence) > 0:
                try:
                    # Filter out nan values
                    valid_confidences = [c for c in confidence if not np.isnan(c)]
                    if valid_confidences:
                        avg_confidence = float(np.mean(valid_confidences))
                except:
                    avg_confidence = 0.0
            
            self.info_label.setText(
                f"Detected {len(boxes)} objects\n"
                f"Classes: {', '.join(set(labels)) if labels else 'None'}\n"
                f"Average confidence: {avg_confidence:.2f}"
            )
            
            # Update stats
            accepted = sum(1 for status in self.review_status.values() if status == 'accept')
            rejected = sum(1 for status in self.review_status.values() if status == 'reject')
            manual = sum(1 for status in self.review_status.values() if status == 'manual')
            
            self.stats_label.setText(
                f"Stats: ✅ {accepted} | ❌ {rejected} | ✏️ {manual}"
            )
            
            # Update button states
            self.prev_btn.setEnabled(self.current_index > 0)
            self.next_btn.setEnabled(self.current_index < len(self.results) - 1)
            
            # Update button colors based on current status
            current_status = self.review_status.get(image_path)
            if current_status == 'accept':
                self.accept_btn.setStyleSheet("background-color: #155724; color: white; font-weight: bold;")
                self.reject_btn.setStyleSheet("background-color: #dc3545; color: white;")
                self.manual_btn.setStyleSheet("background-color: #17a2b8; color: white;")
            elif current_status == 'reject':
                self.accept_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
                self.reject_btn.setStyleSheet("background-color: #8b0000; color: white;")
                self.manual_btn.setStyleSheet("background-color: #17a2b8; color: white;")
            elif current_status == 'manual':
                self.accept_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
                self.reject_btn.setStyleSheet("background-color: #dc3545; color: white;")
                self.manual_btn.setStyleSheet("background-color: #0c5460; color: white;")
            else:
                # Reset to default
                self.accept_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
                self.reject_btn.setStyleSheet("background-color: #dc3545; color: white;")
                self.manual_btn.setStyleSheet("background-color: #17a2b8; color: white;")
    
    def display_image(self, cv_img):
        """Display OpenCV image in QLabel"""
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        
        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
    
    def previous_image(self):
        """Navigate to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
    
    def next_image(self):
        """Navigate to next image"""
        if self.current_index < len(self.results) - 1:
            self.current_index += 1
            self.load_current_image()
    
    def accept_annotation(self):
        """Accept current annotation"""
        if 0 <= self.current_index < len(self.results):
            image_path = self.results[self.current_index][0]
            self.review_status[image_path] = 'accept'
            
            # Auto-move to next if not last
            if self.current_index < len(self.results) - 1:
                self.current_index += 1
                self.load_current_image()
            else:
                self.load_current_image()  # Update button colors
    
    def reject_annotation(self):
        """Reject current annotation"""
        if 0 <= self.current_index < len(self.results):
            image_path = self.results[self.current_index][0]
            self.review_status[image_path] = 'reject'
            self.load_current_image()
    
    def skip_image(self):
        """Skip current image (no decision)"""
        if 0 <= self.current_index < len(self.results):
            if self.current_index < len(self.results) - 1:
                self.current_index += 1
                self.load_current_image()
    
    def edit_manually(self):
        """Open manual editor for current image"""
        if 0 <= self.current_index < len(self.results):
            # Safe unpacking
            result = self.results[self.current_index]
            image_path = result[0]
            boxes = result[1]
            labels = result[2]
            confidence = result[3]
            annotated_img = result[4]
            masks = result[5] if len(result) > 5 else []
            
            # Get class names from labels
            class_names = sorted(set(labels)) if labels else ["object"]
            
            # Open manual annotation dialog
            dialog = ManualAnnotationDialog(
                image_path, boxes, labels, masks, class_names, self
            )
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                # Get edited annotations
                new_boxes, new_labels, new_confidences, new_masks = dialog.get_annotations()
                
                # Update results
                self.results[self.current_index] = (
                    image_path, new_boxes, new_labels, new_confidences, None, new_masks
                )
                
                # Mark as manually edited
                self.review_status[image_path] = 'manual'
                
                # Reload to show updated info
                self.load_current_image()
    
    def save_all_accepted(self):
        """Save all accepted annotations"""
        save_dir = Path("./annotations_accepted")
        save_dir.mkdir(exist_ok=True)
        
        saved_count = 0
        for image_path, boxes, labels, confidences, _, masks in self.results:
            if self.review_status.get(image_path) == 'accept':
                # Save in YOLO format
                self.save_yolo_annotation(
                    image_path, boxes, labels, confidences, masks, save_dir
                )
                saved_count += 1
        
        QMessageBox.information(
            self, "Saved", 
            f"Saved {saved_count} accepted annotations to {save_dir}"
        )
    
    def save_yolo_annotation(self, image_path, boxes, labels, confidences, masks, save_dir):
        """Save annotation in YOLO format"""
        img = cv2.imread(image_path)
        if img is None:
            return
            
        H, W = img.shape[:2]
        
        txt_name = Path(image_path).stem + ".txt"
        txt_path = save_dir / txt_name
        
        with open(txt_path, 'w') as f:
            for i, (box, label) in enumerate(zip(boxes, labels)):
                # Determine ID (default 0 if not found)
                # In a real app we need a consistent class mapping
                class_id = 0 
                # Attempt to map back? For now just use 0 or parse if possible
                
                # Check for mask first
                if masks and i < len(masks) and masks[i] is not None and len(masks[i]) > 0:
                     # Save Polygon: class_id x1 y1 x2 y2 ...
                     # Normalize
                     points = masks[i]
                     # points is (N, 2)
                     normalized_points = []
                     for pt in points:
                         normalized_points.append(f"{pt[0]/W:.6f} {pt[1]/H:.6f}")
                     
                     line = f"{class_id} {' '.join(normalized_points)}\n"
                     f.write(line)
                else:
                    # Save Box: class_id x_center y_center w h
                    x1, y1, x2, y2 = box
                    
                    x_center = ((x1 + x2) / 2) / W
                    y_center = ((y1 + y2) / 2) / H
                    w = (x2 - x1) / W
                    h = (y2 - y1) / H
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        
        # Copy image
        img_save_path = save_dir / Path(image_path).name
        try:
            shutil.copy2(image_path, img_save_path)
        except:
            print(f"Failed to copy image: {image_path}")
    
    def get_final_results(self):
        """Get final results with review status"""
        final_results = []
        for result in self.results:
            image_path = result[0]
            status = self.review_status.get(image_path, 'pending')
            final_results.append((*result, status))
        return final_results

class AIWorker(QThread):
    """Worker thread for AI processing"""
    result_ready = pyqtSignal(object, object, str, list, list, list, list)
    batch_result_ready = pyqtSignal(list)
    progress = pyqtSignal(int)

    def __init__(self, image_paths, prompt, models, settings):
        super().__init__()
        self.image_paths = image_paths if isinstance(image_paths, list) else [image_paths]
        self.prompt = prompt
        self.models = models
        self.settings = settings
        self.batch_mode = len(image_paths) > 1

    def run(self):
        try:
            total = len(self.image_paths)
            all_results = []
            
            for idx, image_path in enumerate(self.image_paths):
                result = self.process_single_image(image_path)
                all_results.append(result)
                
                # Emit progress
                self.progress.emit(int((idx + 1) / total * 100))
                
                # For single image, emit immediately
                if not self.batch_mode:
                    self.result_ready.emit(
                        result['annotated_img'],
                        result['log'],
                        "Done",
                        result['boxes'],
                        result['labels'],
                        result['confidences'],
                        result.get('masks', [])
                    )
            
            # For batch, emit all results
            if self.batch_mode:
                self.batch_result_ready.emit(all_results)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            if not self.batch_mode:
                self.result_ready.emit(None, f"Error: {str(e)}", "Error", [], [], [], [])
            else:
                # Still emit batch results with error info
                error_result = {
                    'annotated_img': None,
                    'log': f"Error: {str(e)}",
                    'boxes': [],
                    'labels': [],
                    'confidences': [],
                    'masks': [],
                    'image_path': self.image_paths[0] if self.image_paths else None
                }
                self.batch_result_ready.emit([error_result])

    def process_single_image(self, image_path):
        """Process a single image"""
        # Parse multiple classes from prompt
        classes = [c.strip() for c in self.prompt.split(',')]
        if not classes:
            classes = ["object"]
        
        # Load Image
        image_cv = cv2.imread(image_path)
        if image_cv is None:
            return {
                'image_path': image_path,
                'annotated_img': None,
                'log': f"❌ Failed to load image: {image_path}",
                'boxes': [],
                'labels': [],
                'confidences': [],
                'masks': []
            }
            
        H, W = image_cv.shape[:2]
        
        log_details = f"📊 Detection for {Path(image_path).name}:\n"
        
        # Handle multiple classes
        all_boxes = []
        all_confidences = []
        all_labels = []
        all_masks = []
        
        for cls in classes:
            cls = cls.strip()
            use_yolo = (self.settings['detection_mode'] == 'yolo' and 
                       YOLO_AVAILABLE and 
                       self.is_yolo_class(cls))
            
            if use_yolo:
                log_details += f"   🚀 Using YOLOv8 for '{cls}'\n"
                raw_boxes, raw_confidences, raw_labels, raw_masks = self.detect_with_yolo(image_cv, cls)
            else:
                log_details += f"   🎯 Using GroundingDINO for '{cls}'\n"
                raw_boxes, raw_confidences, raw_labels, raw_masks = self.detect_with_grounding_dino(image_path, cls)
            
            if len(raw_boxes) > 0:
                all_boxes.extend(raw_boxes)
                all_confidences.extend(raw_confidences)
                all_labels.extend(raw_labels)
                all_masks.extend(raw_masks)
        
        log_details += f"   Raw detections: {len(all_boxes)}\n"

        if len(all_boxes) == 0:
            return {
                'image_path': image_path,
                'annotated_img': None,
                'log': f"❌ No objects detected.\n{log_details}\n💡 Try: Lower confidence or different prompt",
                'boxes': [],
                'labels': [],
                'confidences': [],
                'masks': []
            }

        # Convert to supervision format
        # We need to construct boolean masks from polygons if we want sv.Detections to carry them
        # supervision Expects masks as (N, H, W) np.bool_
        mask_array = None
        if len(all_masks) > 0 and any(m is not None for m in all_masks):
            mask_array = np.zeros((len(all_boxes), H, W), dtype=bool)
            for i, poly in enumerate(all_masks):
                if poly is not None and len(poly) > 0:
                    # poly is list of [x, y], need to convert to int32
                    # cv2.fillPoly expects list of points
                    temp_mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(temp_mask, [poly.astype(np.int32)], 1)
                    mask_array[i] = temp_mask.astype(bool)
        
        detections = sv.Detections(
            xyxy=np.array(all_boxes) if all_boxes else np.array([]).reshape(0, 4),
            confidence=np.array(all_confidences) if all_confidences else np.array([]),
            class_id=np.array([classes.index(lbl) if lbl in classes else 0 for lbl in all_labels]),
            tracker_id=np.arange(len(all_boxes)),
            mask=mask_array
        )

        # Apply NMS to remove overlaps
        if self.settings['use_nms']:
            detections = detections.with_nms(threshold=self.settings['nms_threshold'])
            log_details += f"   After NMS: {len(detections)}\n"

        # Refine with SAM - Now returns masks!
        if self.settings['use_sam_refinement'] and len(detections) > 0:
            # Pass tracker_id to preserve it
            detections, refined_masks = self.refine_with_sam(image_cv, detections, W, H)
            log_details += f"   After SAM refinement: {len(detections)}\n"
            
            # If SAM was used, we should use ITS masks instead of original YOLO masks
            # refined_masks is a list of polygons or binary masks
            # Convert to polygons for drawing/saving if they are binary masks?
            # refine_with_sam now returns valid sv.Detections with masks if possible
            
            # We need to update all_masks with these new high-quality masks
            # However, detections object has been filtered/reordered
            # Simplest way: use the masks returned in detections.mask for drawing overlay
            # AND convert them to polygons for 'final_masks' (saving)
            all_masks = [] # Clear old masks
            if detections.mask is not None:
                 for m in detections.mask:
                     # m is bool HxW
                     # Convert to polygon for consistency with YOLO output format
                     # Find contours
                     contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                     if contours:
                         # Take largest contour
                         c = max(contours, key=cv2.contourArea)
                         # c is (N, 1, 2) -> (N, 2)
                         all_masks.append(c.reshape(-1, 2))
                     else:
                         all_masks.append(None)
            
            # Update tracker_id to match new detections length just in case
            if detections.tracker_id is None:
                 detections.tracker_id = np.arange(len(detections.xyxy))

        # Apply quality filters
        if self.settings['use_filters'] and len(detections) > 0:
            detections = self.apply_quality_filters(detections, W, H)
            log_details += f"   After filtering: {len(detections)}\n"
        
        if len(detections) == 0:
            return {
                'image_path': image_path,
                'annotated_img': None,
                'log': f"❌ No valid objects after filtering.\n{log_details}",
                'boxes': [],
                'labels': [],
                'confidences': [],
                'masks': []
            }

        # Reconstruct masks from tracker_id (for final output polygons)
        final_masks = []
        if detections.tracker_id is not None and len(all_masks) > 0:
            try:
                final_masks = [all_masks[idx] for idx in detections.tracker_id.astype(int) if idx < len(all_masks)]
            except:
                final_masks = [None] * len(detections)
        else:
            final_masks = [None] * len(detections)
            
        # Visualize with class-specific colors
        current_labels = [classes[i] for i in detections.class_id]
        # Pass final_masks (polygons) explicitly to draw_annotations as a robust fallback
        annotated_frame = self.draw_annotations(image_cv, detections, current_labels, classes, final_masks)

        log_msg = f"✅ Detected {len(detections)} objects\n{log_details}"

        return {
            'image_path': image_path,
            'annotated_img': annotated_frame,
            'log': log_msg,
            'boxes': detections.xyxy.tolist(),
            'labels': current_labels,
            'confidences': detections.confidence.tolist(),
            'masks': final_masks
        }

    def is_yolo_class(self, prompt):
        """Check if prompt matches a YOLO class"""
        prompt_lower = prompt.lower().strip()
        # Check exact match or plural
        for yolo_class in YOLO_CLASSES:
            if prompt_lower == yolo_class or prompt_lower == yolo_class + 's':
                return True
        return False

    def detect_with_yolo(self, image_cv, target_class):
        """Detect specific class using YOLOv8"""
        if self.models['yolo'] is None:
            return np.array([]).reshape(0, 4), np.array([]), [], []
        
        try:
            results = self.models['yolo'](image_cv, conf=self.settings['box_threshold'], verbose=False)
        except Exception as e:
            print(f"YOLO detection failed: {e}")
            return np.array([]).reshape(0, 4), np.array([]), [], []
        
        boxes = []
        confidences = []
        labels = []
        masks = []
        
        H, W = image_cv.shape[:2]
        
        for result in results:
            has_masks = hasattr(result, 'masks') and result.masks is not None
            
            for idx, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                class_name = YOLO_CLASSES[cls_id]
                
                # Filter by target class
                if target_class.lower().strip() in [class_name, class_name + 's']:
                    conf = float(box.conf[0])
                    current_mask = None
                    
                    # Use segmentation masks if available
                    if has_masks and self.settings.get('use_yolo_masks', True):
                        try:
                            if hasattr(result.masks, 'xy') and len(result.masks.xy) > idx:
                                polygon = result.masks.xy[idx]
                                if len(polygon) > 0:
                                    current_mask = polygon
                                    x_coords = polygon[:, 0]
                                    y_coords = polygon[:, 1]
                                    
                                    x1 = float(np.min(x_coords))
                                    y1 = float(np.min(y_coords))
                                    x2 = float(np.max(x_coords))
                                    y2 = float(np.max(y_coords))
                                    
                                    # Add padding
                                    pad = 0.02
                                    w_box = x2 - x1
                                    h_box = y2 - y1
                                    x1 = max(0, x1 - w_box * pad)
                                    y1 = max(0, y1 - h_box * pad)
                                    x2 = min(W, x2 + w_box * pad)
                                    y2 = min(H, y2 + h_box * pad)
                                    
                                    xyxy = np.array([x1, y1, x2, y2])
                                else:
                                    xyxy = box.xyxy[0].cpu().numpy()
                            else:
                                # Fallback to box
                                xyxy = box.xyxy[0].cpu().numpy()
                        except Exception as e:
                            print(f"YOLO mask extraction failed: {e}")
                            xyxy = box.xyxy[0].cpu().numpy()
                    else:
                        xyxy = box.xyxy[0].cpu().numpy()
                    
                    boxes.append(xyxy)
                    confidences.append(conf)
                    labels.append(class_name)
                    masks.append(current_mask)
        
        return (np.array(boxes) if boxes else np.array([]).reshape(0, 4),
                np.array(confidences) if confidences else np.array([]),
                labels,
                masks)

    def detect_with_grounding_dino(self, image_path, target_class):
        """Detect specific class using GroundingDINO"""
        try:
            image_source, image = load_image(image_path)
        except Exception as e:
            print(f"Failed to load image for GroundingDINO: {e}")
            return np.array([]).reshape(0, 4), np.array([]), [], []
        
        if hasattr(image_source, 'height'):
            H, W = image_source.height, image_source.width
        else:
            H, W = image_source.shape[:2]
        
        all_boxes = []
        all_confidences = []
        all_labels = []
        all_masks = []
        
        thresholds = [(self.settings['box_threshold'], self.settings['text_threshold'])]
        if self.settings['use_multi_threshold']:
            lower_thresh = max(0.15, self.settings['box_threshold'] - 0.10)
            thresholds.append((lower_thresh, self.settings['text_threshold']))
        
        for box_th, text_th in thresholds:
            try:
                boxes, logits, phrases = predict(
                    model=self.models['grounding_dino'],
                    image=image,
                    caption=target_class,
                    box_threshold=box_th,
                    text_threshold=text_th,
                    device=DEVICE
                )
                
                if len(boxes) > 0:
                    boxes_xyxy = sv.xcycwh_to_xyxy(boxes.numpy()) * np.array([W, H, W, H])
                    all_boxes.append(boxes_xyxy)
                    all_confidences.append(logits.numpy())
                    all_labels.extend([target_class] * len(boxes))
                    # GroundingDINO doesn't return masks natively
                    all_masks.extend([None] * len(boxes))
            except Exception as e:
                print(f"GroundingDINO prediction failed: {e}")
                continue
        
        if all_boxes:
            return (np.vstack(all_boxes),
                   np.concatenate(all_confidences),
                   all_labels,
                   all_masks)
        else:
            return np.array([]).reshape(0, 4), np.array([]), [], []

    def refine_with_sam(self, image_cv, detections, W, H):
        """Refine boxes AND generate masks using SAM segmentation"""
        if len(detections) == 0:
            return detections, []
        
        try:
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            self.models['sam_predictor'].set_image(image_rgb)
        except Exception as e:
            print(f"SAM image processing failed: {e}")
            return detections, []
        
        refined_boxes = []
        valid_confidences = []
        valid_class_ids = []
        valid_tracker_ids = []
        generated_masks = [] # Bool masks (H, W)
        
        for i, box in enumerate(detections.xyxy):
            try:
                x1, y1, x2, y2 = box
                
                # Use multiple points for better coverage
                h_box = y2 - y1
                w_box = x2 - x1
                
                input_box = np.array([x1, y1, x2, y2])
                
                # SAM Inference with Box Prompt
                masks, scores, _ = self.models['sam_predictor'].predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False, # Single best mask
                )
                
                # mask is (1, H, W)
                best_mask = masks[0]
                generated_masks.append(best_mask)
                
                # Convert mask to bounding box
                y_indices, x_indices = np.where(best_mask)
                
                current_box = box
                if len(x_indices) > 0:
                    x1_new = float(np.min(x_indices))
                    y1_new = float(np.min(y_indices))
                    x2_new = float(np.max(x_indices))
                    y2_new = float(np.max(y_indices))
                    
                    # Update box if valid
                    current_box = [x1_new, y1_new, x2_new, y2_new]
                
                refined_boxes.append(current_box)
                valid_confidences.append(detections.confidence[i])
                valid_class_ids.append(detections.class_id[i])
                if detections.tracker_id is not None:
                     valid_tracker_ids.append(detections.tracker_id[i])
                else:
                     valid_tracker_ids.append(i)
                
            except Exception as e:
                print(f"SAM refinement failed for box {i}: {e}")
                refined_boxes.append(box)
                valid_confidences.append(detections.confidence[i])
                valid_class_ids.append(detections.class_id[i])
                generated_masks.append(np.zeros((H, W), dtype=bool)) # Empty mask fallback
                if detections.tracker_id is not None:
                     valid_tracker_ids.append(detections.tracker_id[i])
        
        # Stack masks
        if generated_masks:
            mask_stack = np.stack(generated_masks)
        else:
            mask_stack = None

        new_detections = sv.Detections(
            xyxy=np.array(refined_boxes),
            confidence=np.array(valid_confidences),
            class_id=np.array(valid_class_ids),
            tracker_id=np.array(valid_tracker_ids),
            mask=mask_stack
        )
        
        return new_detections, generated_masks
        
        

    def apply_quality_filters(self, detections, W, H):
        """Apply quality filters"""
        if len(detections) == 0:
            return detections
        
        valid_mask = np.ones(len(detections), dtype=bool)
        
        for i, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = box
            width = (x2 - x1) / W
            height = (y2 - y1) / H
            area = width * height
            
            # Min size filter
            if self.settings['min_size_filter'] and area < self.settings['min_area']:
                valid_mask[i] = False
                continue
            
            # Max size filter
            if self.settings['max_size_filter'] and area > self.settings['max_area']:
                valid_mask[i] = False
                continue
        
        return detections[valid_mask]

    def draw_annotations(self, image_cv, detections, labels, classes, polygons=None):
        """Draw bounding boxes and masks with class-specific colors"""
        annotated = image_cv.copy()
        overlay = annotated.copy()
        
        has_polygons = polygons is not None and len(polygons) == len(detections)
        
        # Create color palette for each class
        colors = {}
        for i, cls in enumerate(classes):
            # Generate distinct colors
            hue = (i * 137) % 360  # Golden angle approximation
            color = QColor.fromHsv(hue, 255, 255)
            # OpenCV uses BGR
            colors[i] = (color.blue(), color.green(), color.red())
        
        # Check for masks
        has_masks = hasattr(detections, 'mask') and detections.mask is not None
        # Usually supervision stores masks in detections.mask as boolean array (N, H, W) or specific format
        # However, our propagation logic might rely on how it was constructed or if we passed it differently.
        # But if we used SAM, we want to visualize what's in detections.mask if available.
        
        for i, (box, label, conf) in enumerate(zip(detections.xyxy, labels, detections.confidence)):
            x1, y1, x2, y2 = box.astype(int)
            class_id = classes.index(label) if label in classes else 0
            color = colors.get(class_id, (0, 255, 0))
            
            # Draw Mask if available
            # Prioritize polygons if passed (since they are direct from YOLO)
            if has_polygons and polygons[i] is not None:
                # Draw using polygon (faster/cleaner for drawing)
                poly = polygons[i].astype(np.int32)
                cv2.fillPoly(overlay, [poly], color)
                
            elif has_masks:
                mask = detections.mask[i]
                if mask is not None:
                    # Draw mask overlay
                    # supervision masks are bool HxW
                    if isinstance(mask, np.ndarray) and mask.ndim == 2:
                        # Apply to overlay using boolean indexing
                        overlay[mask] = color

            # Draw rectangle (thicker for visibility)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
            
            # Draw label with smaller font
            label_text = f"{label} {conf:.2f}"
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Draw background for text
            cv2.rectangle(
                annotated, 
                (x1, y1 - text_height - 5), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated, 
                label_text, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (255, 255, 255), 
                thickness
            )
        
        # Blend overlay for transparency
        alpha = 0.45  # 45% transparency for masks - Dark/Visible enough
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
        
        return annotated

class UniversalAnnotateApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Universal Auto-Annotation Tool - YOLOv8 + SAM")
        self.resize(1280, 720) # Use resize instead of setGeometry for better default behavior
        
        # Application state
        self.current_image_paths = []
        self.batch_results = []
        self.reviewed_results = []
        
        # Create main layout
        self.init_ui()
        
        # Load Models
        self.log("⏳ Loading AI Models...")
        self.models = {}
        
        # Load GroundingDINO
        try:
            self.models['grounding_dino'] = load_model(GD_CONFIG, GD_CHECKPOINT, device=DEVICE)
            self.log("✅ GroundingDINO loaded")
        except Exception as e:
            self.log(f"❌ Failed to load GroundingDINO: {e}")
            self.models['grounding_dino'] = None
        
        # Load YOLOv8
        if YOLO_AVAILABLE and os.path.exists(YOLO_CHECKPOINT):
            try:
                self.models['yolo'] = YOLO(YOLO_CHECKPOINT)
                self.log("✅ YOLOv8 loaded")
            except Exception as e:
                self.log(f"❌ Failed to load YOLOv8: {e}")
                self.models['yolo'] = None
        else:
            self.models['yolo'] = None
            self.log("⚠️  YOLOv8 not available")
        
        # Load SAM
        try:
            sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
            sam.to(device=DEVICE)
            self.models['sam_predictor'] = SamPredictor(sam)
            self.log("✅ SAM loaded")
        except Exception as e:
            self.log(f"❌ Failed to load SAM: {e}")
            self.models['sam_predictor'] = None
        
        self.log(f"✅ All models ready on {DEVICE}")

    def init_ui(self):
        """Initialize the user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Use QSplitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel (controls) wrapped in ScrollArea
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        left_panel = self.create_control_panel()
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(300)
        
        splitter.addWidget(left_scroll)
        
        # Right panel (display)
        right_panel = self.create_display_panel()
        splitter.addWidget(right_panel)
        
        # Set initial sizes (propotional)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(splitter)
        
        # Create menu bar
        self.create_menu_bar()

    def create_menu_bar(self):
        """Create menu bar with additional features"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_action = QAction("Load Image", self)
        load_action.triggered.connect(self.load_single_image)
        load_action.setShortcut("Ctrl+O")
        file_menu.addAction(load_action)
        
        load_folder_action = QAction("Load Folder", self)
        load_folder_action.triggered.connect(self.load_batch_images)
        load_folder_action.setShortcut("Ctrl+Shift+O")
        file_menu.addAction(load_folder_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("Export Annotations", self)
        export_action.triggered.connect(self.export_annotations)
        export_action.setShortcut("Ctrl+S")
        file_menu.addAction(export_action)
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        manual_annotate_action = QAction("Manual Annotation Tool", self)
        manual_annotate_action.triggered.connect(self.open_manual_annotation)
        manual_annotate_action.setShortcut("Ctrl+M")
        edit_menu.addAction(manual_annotate_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        batch_review_action = QAction("Batch Review Tool", self)
        batch_review_action.triggered.connect(self.open_batch_review)
        batch_review_action.setShortcut("Ctrl+B")
        tools_menu.addAction(batch_review_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_control_panel(self):
        """Create the control panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("🎯 Auto-Annotation Controls")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Detection Mode
        mode_group = QGroupBox("Detection Method")
        mode_layout = QVBoxLayout()
        
        self.radio_yolo = QRadioButton("YOLOv8 (Fast & Accurate for 80 common objects)")
        self.radio_grounding = QRadioButton("GroundingDINO (Any custom prompt)")
        self.radio_yolo.setChecked(True)
        
        mode_layout.addWidget(self.radio_yolo)
        mode_layout.addWidget(self.radio_grounding)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Prompt Input
        prompt_group = QGroupBox("Detection Target")
        prompt_layout = QVBoxLayout()
        
        self.prompt_input = QLineEdit("person")
        self.prompt_input.setPlaceholderText("Enter comma-separated classes (e.g., person,car,dog)")
        prompt_layout.addWidget(QLabel("Enter object(s) to detect:"))
        prompt_layout.addWidget(self.prompt_input)
        
        # Example prompts
        example_label = QLabel("Examples: 'person,car' or 'cat,dog'")
        example_label.setStyleSheet("font-size: 10px; color: #888;")
        prompt_layout.addWidget(example_label)
        
        # YOLO class buttons
        preset_layout = QGridLayout()
        presets = ["person", "car", "dog", "cat", "bottle", "chair", "tv", "book"]
        for i, preset in enumerate(presets):
            btn = QPushButton(preset)
            btn.clicked.connect(lambda checked, p=preset: self.add_to_prompt(p))
            preset_layout.addWidget(btn, i // 4, i % 4)
        
        prompt_layout.addLayout(preset_layout)
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        
        # Detection Settings
        settings_group = QGroupBox("⚙️ Detection Settings")
        settings_layout = QVBoxLayout()
        
        settings_layout.addWidget(QLabel("Confidence Threshold:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(10)
        self.conf_slider.setMaximum(70)
        self.conf_slider.setValue(25)
        self.conf_label = QLabel("0.25")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_label.setText(f"{v/100:.2f}")
        )
        settings_layout.addWidget(self.conf_slider)
        settings_layout.addWidget(self.conf_label)
        
        self.nms_check = QCheckBox("Use NMS (remove overlaps)")
        self.nms_check.setChecked(True)
        settings_layout.addWidget(self.nms_check)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # SAM Refinement
        sam_group = QGroupBox("🎯 Additional Refinement")
        sam_layout = QVBoxLayout()
        
        self.sam_check = QCheckBox("Enable SAM refinement")
        self.sam_check.setChecked(False)
        sam_layout.addWidget(self.sam_check)
        
        sam_info = QLabel("💡 Helps with complex shapes")
        sam_info.setStyleSheet("font-size: 10px; color: #888;")
        sam_layout.addWidget(sam_info)
        
        sam_group.setLayout(sam_layout)
        layout.addWidget(sam_group)
        
        # Filters
        filter_group = QGroupBox("🔍 Quality Filters")
        filter_layout = QVBoxLayout()
        
        self.filter_check = QCheckBox("Enable filters")
        self.filter_check.setChecked(True)
        filter_layout.addWidget(self.filter_check)
        
        filter_layout.addWidget(QLabel("Min Area (% of image):"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setMinimum(1)
        self.min_area_spin.setMaximum(100)
        self.min_area_spin.setValue(2)
        filter_layout.addWidget(self.min_area_spin)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Action Buttons
        self.btn_load_single = QPushButton("📂 Load Single Image")
        self.btn_load_batch = QPushButton("📁 Load Batch Folder")
        self.btn_run = QPushButton("▶️ Run Detection")
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet("""
            font-size: 16px; 
            font-weight: bold; 
            padding: 10px; 
            background-color: #28a745; 
            color: white;
            border-radius: 5px;
        """)
        
        layout.addWidget(self.btn_load_single)
        layout.addWidget(self.btn_load_batch)
        layout.addWidget(self.btn_run)
        
        # Manual annotation button
        self.btn_manual = QPushButton("✏️ Open Manual Editor")
        self.btn_manual.setEnabled(False)
        self.btn_manual.setStyleSheet("""
            padding: 8px;
            background-color: #17a2b8;
            color: white;
            border-radius: 5px;
        """)
        layout.addWidget(self.btn_manual)
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Auto-save options
        save_group = QGroupBox("💾 Save Options")
        save_layout = QVBoxLayout()
        
        self.auto_save_check = QCheckBox("Auto-save annotations (YOLO format)")
        self.auto_save_check.setChecked(True)
        save_layout.addWidget(self.auto_save_check)
        
        self.save_for_finetuning = QCheckBox("Save for fine-tuning (with permission)")
        self.save_for_finetuning.setChecked(False)
        save_layout.addWidget(self.save_for_finetuning)
        
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
        
        # Connect signals
        self.btn_load_single.clicked.connect(self.load_single_image)
        self.btn_load_batch.clicked.connect(self.load_batch_images)
        self.btn_run.clicked.connect(self.run_detection)
        self.btn_manual.clicked.connect(self.open_manual_annotation)
        
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel

    def create_display_panel(self):
        """Create the display panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Image display
        self.image_label = QLabel("Load Image to Start")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            border: 2px solid #444; 
            background-color: #222;
            border-radius: 5px;
        """)
        layout.addWidget(self.image_label, stretch=3)
        
        # Batch controls (hidden by default)
        self.batch_controls = QWidget()
        batch_layout = QHBoxLayout()
        
        self.batch_accept_btn = QPushButton("✅ Accept & Next")
        self.batch_accept_btn.clicked.connect(self.accept_and_next)
        self.batch_accept_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        
        self.batch_reject_btn = QPushButton("❌ Reject & Edit")
        self.batch_reject_btn.clicked.connect(self.reject_and_edit)
        self.batch_reject_btn.setStyleSheet("background-color: #dc3545; color: white;")
        
        self.batch_skip_btn = QPushButton("⏭ Skip")
        self.batch_skip_btn.clicked.connect(self.skip_image)
        self.batch_skip_btn.setStyleSheet("background-color: #6c757d; color: white;")
        
        batch_layout.addWidget(self.batch_accept_btn)
        batch_layout.addWidget(self.batch_reject_btn)
        batch_layout.addWidget(self.batch_skip_btn)
        
        self.batch_controls.setLayout(batch_layout)
        self.batch_controls.setVisible(False)
        layout.addWidget(self.batch_controls)
        
        # Log box
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(100) # Ensure it has minimum height but can grow
        self.log_box.setStyleSheet("""
            background-color: #111; 
            color: #0f0; 
            font-family: Monospace;
            border-radius: 5px;
        """)
        layout.addWidget(self.log_box, stretch=0) # Don't stretch log infinitely
        
        panel.setLayout(layout)
        return panel

    def log(self, msg):
        """Add message to log"""
        self.log_box.append(msg)
        # Auto-scroll to bottom
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )

    def add_to_prompt(self, text):
        """Add text to prompt input"""
        current = self.prompt_input.text()
        if current:
            self.prompt_input.setText(f"{current},{text}")
        else:
            self.prompt_input.setText(text)

    def load_single_image(self):
        """Load a single image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.jpg *.png *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            self.current_image_paths = [file_path]
            self.show_image(cv2.imread(file_path))
            self.btn_run.setEnabled(True)
            self.btn_manual.setEnabled(True)
            self.log(f"📷 Loaded: {Path(file_path).name}")

    def load_batch_images(self):
        """Load multiple images from a folder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.current_image_paths = []
            extensions = ['*.jpg', '*.png', '*.jpeg', '*.bmp', '*.tiff']
            
            for ext in extensions:
                self.current_image_paths.extend(Path(folder_path).glob(ext))
            
            self.current_image_paths = [str(p) for p in self.current_image_paths]
            
            if self.current_image_paths:
                self.show_image(cv2.imread(self.current_image_paths[0]))
                self.btn_run.setEnabled(True)
                self.btn_manual.setEnabled(True)
                self.log(f"📁 Loaded {len(self.current_image_paths)} images")
                
                # Show batch controls
                if len(self.current_image_paths) > 1:
                    self.batch_controls.setVisible(True)
            else:
                self.log("❌ No images found")

    def show_image(self, cv_img):
        """Display OpenCV image in QLabel"""
        if cv_img is None:
            self.image_label.setText("Failed to load image")
            return
        
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        
        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def run_detection(self):
        """Run detection on current image(s)"""
        prompt = self.prompt_input.text().strip()
        if not prompt:
            self.log("❌ Please enter a prompt")
            return
        
        # Check if models are loaded
        if self.radio_yolo.isChecked() and self.models.get('yolo') is None:
            self.log("❌ YOLOv8 model not loaded")
            return
        elif self.radio_grounding.isChecked() and self.models.get('grounding_dino') is None:
            self.log("❌ GroundingDINO model not loaded")
            return
        
        settings = {
            'detection_mode': 'yolo' if self.radio_yolo.isChecked() else 'grounding_dino',
            'box_threshold': self.conf_slider.value() / 100.0,
            'text_threshold': 0.25,
            'nms_threshold': 0.50,
            'use_nms': self.nms_check.isChecked(),
            'use_multi_threshold': False,
            'use_yolo_masks': True,
            'use_sam_refinement': self.sam_check.isChecked(),
            'use_filters': self.filter_check.isChecked(),
            'min_size_filter': True,
            'max_size_filter': True,
            'min_area': self.min_area_spin.value() / 1000.0,
            'max_area': 0.95,
            'auto_save': self.auto_save_check.isChecked(),
            'output_dir': './annotations'
        }
        
        self.log(f"🚀 Processing with: {settings['detection_mode'].upper()}")
        self.log(f"   Target(s): '{prompt}' | Conf: {settings['box_threshold']:.2f}")
        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.worker = AIWorker(self.current_image_paths, prompt, self.models, settings)
        
        if len(self.current_image_paths) > 1:
            # Batch processing
            self.worker.batch_result_ready.connect(self.on_batch_detection_finished)
        else:
            # Single image
            self.worker.result_ready.connect(self.on_single_detection_finished)
            
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.start()

    def on_single_detection_finished(self, img, log, status, boxes, labels, confidences, masks=None):
        """Handle single image detection finished"""
        self.btn_run.setEnabled(True)
        if img is not None:
            self.show_image(img)
        self.log(log)
        self.log(f"✅ Complete! Found {len(boxes)} objects")
        
        # Store result for possible manual editing
        if len(self.current_image_paths) == 1:
            self.current_result = {
                'image_path': self.current_image_paths[0],
                'boxes': boxes,
                'labels': labels,
                'confidences': confidences,
                'masks': masks,
                'annotated_img': img
            }

    def on_batch_detection_finished(self, results):
        """Handle batch detection finished"""
        self.btn_run.setEnabled(True)
        self.batch_results = results
        
        # Filter out failed results
        valid_results = [r for r in results if r.get('annotated_img') is not None]
        
        if valid_results:
            # Show first result
            self.show_image(valid_results[0]['annotated_img'])
            self.log(valid_results[0]['log'])
            
            # Ask user if they want to review batch
            reply = QMessageBox.question(
                self, 'Batch Processing Complete',
                f'Processed {len(valid_results)} images. Would you like to review them?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.open_batch_review()
            else:
                # Auto-save all if enabled
                if self.auto_save_check.isChecked():
                    self.save_all_batch_results(valid_results)
        else:
            self.log("❌ No valid results generated")
            QMessageBox.warning(self, "No Results", "No valid annotations were generated.")

    def open_manual_annotation(self):
        """Open manual annotation editor"""
        if not self.current_image_paths:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        
        try:
            # Get current annotations if available
            initial_boxes = []
            initial_labels = []
            initial_masks = []
            
            if hasattr(self, 'current_result'):
                initial_boxes = self.current_result.get('boxes', [])
                initial_labels = self.current_result.get('labels', [])
                initial_masks = self.current_result.get('masks', [])
            
            # Parse classes from prompt
            prompt_text = self.prompt_input.text().strip()
            classes = [c.strip() for c in prompt_text.split(',')] if prompt_text else ["object"]
            
            dialog = ManualAnnotationDialog(
                self.current_image_paths[0],
                initial_boxes,
                initial_labels,
                initial_masks,
                classes,
                self
            )
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                # Get manual annotations
                boxes, labels, confidences, masks = dialog.get_annotations()
                
                # Update display
                if boxes:
                    # Create annotated image
                    img = cv2.imread(self.current_image_paths[0])
                    annotated_img = self.draw_manual_annotations(img, boxes, labels)
                    self.show_image(annotated_img)
                    
                    # Update current result
                    self.current_result = {
                        'image_path': self.current_image_paths[0],
                        'boxes': boxes,
                        'labels': labels,
                        'confidences': confidences,
                        'annotated_img': annotated_img
                    }
                    
                    self.log(f"✏️ Manual annotation saved: {len(boxes)} objects")
                    
                    # Ask about saving for fine-tuning
                    if self.save_for_finetuning.isChecked():
                        self.ask_finetuning_permission()
                else:
                    self.log("⚠️ No annotations created")
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open manual editor: {str(e)}")

    def draw_manual_annotations(self, img, boxes, labels):
        """Draw manual annotations on image"""
        annotated = img.copy()
        
        # Create color palette
        unique_labels = list(set(labels))
        colors = {}
        for i, label in enumerate(unique_labels):
            hue = (i * 137) % 360
            color = QColor.fromHsv(hue, 255, 255)
            colors[label] = (color.red(), color.green(), color.blue())
        
        # Draw boxes
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            color = colors.get(label, (0, 255, 0))
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with smaller font
            label_text = f"{label}"
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Draw background for text
            cv2.rectangle(
                annotated, 
                (x1, y1 - text_height - 5), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated, 
                label_text, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (255, 255, 255), 
                thickness
            )
        
        return annotated

    def open_batch_review(self):
        """Open batch review dialog"""
        if not self.batch_results:
            QMessageBox.warning(self, "No Results", "Please run batch detection first.")
            return
        
        # Filter only valid results
        valid_results = []
        for result in self.batch_results:
            if result.get('annotated_img') is not None:
                valid_results.append(result)
        
        if not valid_results:
            QMessageBox.warning(self, "No Valid Results", "No valid annotations to review.")
            return
        
        # Prepare results for review
        review_data = []
        for result in valid_results:
            review_data.append((
                result['image_path'],
                result['boxes'],
                result['labels'],
                result['confidences'],
                result['annotated_img'],
                result.get('masks', [])
            ))
        
        dialog = BatchReviewDialog(review_data, self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get reviewed results
            final_results = dialog.get_final_results()
            self.reviewed_results = final_results
            
            # Save accepted results
            accepted_count = sum(1 for r in final_results if r[-1] == 'accept')
            if accepted_count > 0:
                self.log(f"✅ Accepted {accepted_count} images")
                
                # Ask about saving for fine-tuning
                if self.save_for_finetuning.isChecked():
                    self.ask_finetuning_permission()
            else:
                self.log("⚠️ No images were accepted")

    def accept_and_next(self):
        """Accept current annotation and move to next in batch"""
        if not self.batch_results:
            return
        
        # Find current image index
        current_img = self.current_image_paths[0] if self.current_image_paths else None
        if not current_img:
            return
        
        current_index = next((i for i, r in enumerate(self.batch_results) 
                            if r['image_path'] == current_img), -1)
        
        if current_index >= 0:
            self.log(f"✅ Accepted image {current_index + 1}/{len(self.batch_results)}")
            
            # Move to next image
            next_index = current_index + 1
            if next_index < len(self.batch_results):
                next_result = self.batch_results[next_index]
                if next_result.get('annotated_img') is not None:
                    self.show_image(next_result['annotated_img'])
                    self.log(next_result['log'])
                    
                    # Update current paths
                    self.current_image_paths = [next_result['image_path']]
                else:
                    # Skip to next valid image
                    self.skip_image()
            else:
                self.log("🏁 Finished reviewing all images")
                self.batch_controls.setVisible(False)

    def reject_and_edit(self):
        """Reject current annotation and open manual editor"""
        if not self.current_image_paths:
            return
        
        self.open_manual_annotation()

    def skip_image(self):
        """Skip current image in batch"""
        if not self.batch_results:
            return
        
        current_img = self.current_image_paths[0] if self.current_image_paths else None
        if not current_img:
            return
        
        current_index = next((i for i, r in enumerate(self.batch_results) 
                            if r['image_path'] == current_img), -1)
        
        if current_index >= 0:
            self.log(f"⏭ Skipped image {current_index + 1}/{len(self.batch_results)}")
            
            # Move to next valid image
            next_index = current_index + 1
            while next_index < len(self.batch_results):
                next_result = self.batch_results[next_index]
                if next_result.get('annotated_img') is not None:
                    self.show_image(next_result['annotated_img'])
                    self.log(next_result['log'])
                    self.current_image_paths = [next_result['image_path']]
                    break
                next_index += 1
            else:
                self.log("🏁 Finished reviewing all images")
                self.batch_controls.setVisible(False)

    def ask_finetuning_permission(self):
        """Ask user for permission to save data for fine-tuning"""
        reply = QMessageBox.question(
            self, 'Fine-tuning Permission',
            'Would you like to save these annotations for fine-tuning the model?\n\n'
            'This will save images and annotations in a format suitable for training.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.save_for_finetuning_data()

    def save_for_finetuning_data(self):
        """Save data for fine-tuning"""
        save_dir = Path("./finetuning_data")
        save_dir.mkdir(exist_ok=True)
        
        # Create YOLO dataset structure
        images_dir = save_dir / "images"
        labels_dir = save_dir / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        saved_count = 0
        results_to_save = self.reviewed_results if self.reviewed_results else self.batch_results
        
        for result in results_to_save:
            if isinstance(result, dict):
                # Single result format
                image_path = result['image_path']
                boxes = result['boxes']
                labels = result['labels']
                status = 'accept'  # Assume accepted for single results
            else:
                # Reviewed result format
                image_path, boxes, labels, _, _, status = result
                if status != 'accept':
                    continue
            
            # Skip if no boxes
            if not boxes or len(boxes) == 0:
                continue
            
            # Copy image
            img_name = Path(image_path).name
            img_save_path = images_dir / img_name
            try:
                shutil.copy2(image_path, img_save_path)
            except Exception as e:
                print(f"Failed to copy image: {e}")
                continue
            
            # Save annotations
            img = cv2.imread(image_path)
            if img is None:
                continue
                
            H, W = img.shape[:2]
            
            # Create class mapping
            unique_labels = sorted(set(labels))
            class_to_id = {cls: i for i, cls in enumerate(unique_labels)}
            
            # Save class names
            try:
                with open(save_dir / "classes.txt", 'w') as f:
                    for cls in unique_labels:
                        f.write(f"{cls}\n")
            except Exception as e:
                print(f"Failed to save classes.txt: {e}")
                continue
            
            # Save annotations
            txt_path = labels_dir / f"{Path(image_path).stem}.txt"
            try:
                with open(txt_path, 'w') as f:
                    for box, label in zip(boxes, labels):
                        x1, y1, x2, y2 = box
                        x_center = ((x1 + x2) / 2) / W
                        y_center = ((y1 + y2) / 2) / H
                        width = (x2 - x1) / W
                        height = (y2 - y1) / H
                        
                        class_id = class_to_id.get(label, 0)
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            except Exception as e:
                print(f"Failed to save annotation file: {e}")
                continue
            
            saved_count += 1
        
        if saved_count > 0:
            self.log(f"💾 Saved {saved_count} images for fine-tuning in {save_dir}")
            
            # Create dataset.yaml file
            try:
                unique_labels_list = []
                for result in results_to_save:
                    if isinstance(result, dict):
                        unique_labels_list.extend(result['labels'])
                    else:
                        unique_labels_list.extend(result[2])  # labels are at index 2
                
                unique_labels = sorted(set(unique_labels_list))
                
                yaml_content = f"""path: {save_dir.absolute()}
train: images
val: images

nc: {len(unique_labels)}
names: {list(unique_labels)}
"""
                
                with open(save_dir / "dataset.yaml", 'w') as f:
                    f.write(yaml_content)
                
                QMessageBox.information(
                    self, "Saved for Fine-tuning",
                    f"Saved {saved_count} images and annotations to {save_dir}\n\n"
                    f"Dataset is ready for YOLO training. Use the dataset.yaml file."
                )
            except Exception as e:
                self.log(f"⚠️ Failed to create dataset.yaml: {e}")
        else:
            self.log("⚠️ No valid annotations to save for fine-tuning")

    def save_all_batch_results(self, results):
        """Save all batch results automatically"""
        if not results:
            return
        
        save_dir = Path("./batch_annotations")
        save_dir.mkdir(exist_ok=True)
        
        saved_count = 0
        for result in results:
            if result.get('annotated_img') is not None:
                try:
                    # Save annotated image
                    img_name = Path(result['image_path']).stem + "_annotated.jpg"
                    img_save_path = save_dir / img_name
                    cv2.imwrite(str(img_save_path), result['annotated_img'])
                    
                    # Save annotations
                    txt_path = save_dir / f"{Path(result['image_path']).stem}.txt"
                    with open(txt_path, 'w') as f:
                        boxes = result.get('boxes', [])
                        labels = result.get('labels', [])
                        confidences = result.get('confidences', [])
                        
                        for box, label, conf in zip(boxes, labels, confidences):
                            f.write(f"{label} {box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f} {conf:.3f}\n")
                    
                    saved_count += 1
                except Exception as e:
                    print(f"Failed to save batch result: {e}")
        
        self.log(f"💾 Auto-saved {saved_count} batch annotations to {save_dir}")

    def export_annotations(self):
        """Export annotations to various formats"""
        if not hasattr(self, 'current_result') or not self.current_result:
            QMessageBox.warning(self, "No Data", "No annotations to export.")
            return
        
        formats = "YOLO (*.txt);;COCO (*.json);;Pascal VOC (*.xml);;All Files (*)"
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Annotations", "", formats
        )
        
        if file_path:
            try:
                if selected_filter == "YOLO (*.txt)":
                    self.export_yolo_format(file_path)
                elif selected_filter == "COCO (*.json)":
                    self.export_coco_format(file_path)
                elif selected_filter == "Pascal VOC (*.xml)":
                    self.export_voc_format(file_path)
                
                self.log(f"📤 Exported annotations to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

    def export_yolo_format(self, file_path):
        """Export in YOLO format"""
        result = self.current_result
        img = cv2.imread(result['image_path'])
        H, W = img.shape[:2]
        
        with open(file_path, 'w') as f:
            unique_labels = sorted(set(result['labels']))
            class_to_id = {cls: i for i, cls in enumerate(unique_labels)}
            
            for box, label in zip(result['boxes'], result['labels']):
                x1, y1, x2, y2 = box
                x_center = ((x1 + x2) / 2) / W
                y_center = ((y1 + y2) / 2) / H
                width = (x2 - x1) / W
                height = (y2 - y1) / H
                
                class_id = class_to_id[label]
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def export_coco_format(self, file_path):
        """Export in COCO format (simplified)"""
        result = self.current_result
        img = cv2.imread(result['image_path'])
        H, W = img.shape[:2]
        
        coco_data = {
            "info": {"description": "Auto-generated annotations"},
            "images": [{
                "id": 0,
                "file_name": Path(result['image_path']).name,
                "height": H,
                "width": W
            }],
            "annotations": [],
            "categories": []
        }
        
        unique_labels = sorted(set(result['labels']))
        for i, label in enumerate(unique_labels):
            coco_data["categories"].append({
                "id": i,
                "name": label,
                "supercategory": "object"
            })
        
        class_to_id = {cls: i for i, cls in enumerate(unique_labels)}
        
        for i, (box, label) in enumerate(zip(result['boxes'], result['labels'])):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            coco_data["annotations"].append({
                "id": i,
                "image_id": 0,
                "category_id": class_to_id[label],
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "segmentation": [],
                "iscrowd": 0
            })
        
        with open(file_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

    def export_voc_format(self, file_path):
        """Export in Pascal VOC format (simplified)"""
        from xml.etree.ElementTree import Element, SubElement, tostring
        from xml.dom import minidom
        
        result = self.current_result
        img = cv2.imread(result['image_path'])
        H, W = img.shape[:2]
        
        root = Element('annotation')
        
        # Basic info
        SubElement(root, 'filename').text = Path(result['image_path']).name
        SubElement(root, 'folder').text = 'annotations'
        
        size = SubElement(root, 'size')
        SubElement(size, 'width').text = str(W)
        SubElement(size, 'height').text = str(H)
        SubElement(size, 'depth').text = '3'
        
        # Objects
        for box, label in zip(result['boxes'], result['labels']):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            obj = SubElement(root, 'object')
            SubElement(obj, 'name').text = label
            SubElement(obj, 'pose').text = 'Unspecified'
            SubElement(obj, 'truncated').text = '0'
            SubElement(obj, 'difficult').text = '0'
            
            bndbox = SubElement(obj, 'bndbox')
            SubElement(bndbox, 'xmin').text = str(x1)
            SubElement(bndbox, 'ymin').text = str(y1)
            SubElement(bndbox, 'xmax').text = str(x2)
            SubElement(bndbox, 'ymax').text = str(y2)
        
        # Pretty print
        xml_str = minidom.parseString(tostring(root)).toprettyxml(indent="  ")
        with open(file_path, 'w') as f:
            f.write(xml_str)

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "Universal Auto-Annotation Tool",
            f"""<h2>Universal Auto-Annotation Tool v2.0</h2>
            <p>An intelligent annotation tool combining YOLOv8, GroundingDINO, and SAM.</p>
            
            <h3>Features:</h3>
            <ul>
                <li>Multiple detection methods (YOLOv8 & GroundingDINO)</li>
                <li>Multi-class detection support</li>
                <li>Manual annotation editor</li>
                <li>Batch review with accept/reject system</li>
                <li>Export to multiple formats (YOLO, COCO, VOC)</li>
                <li>Fine-tuning data collection</li>
                <li>Real-time visualization</li>
            </ul>
            
            <p><b>Device:</b> {DEVICE.upper()}</p>
            <p><b>YOLO Available:</b> {YOLO_AVAILABLE}</p>
            
            <p>© 2024 Universal Auto-Annotation Tool</p>
            """
        )


if __name__ == "__main__":
    # Set application style
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    
    # Set palette
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(palette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(palette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(palette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(palette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(palette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(palette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(palette.ColorRole.Highlight, QColor(142, 45, 197).lighter())
    palette.setColor(palette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)
    
    window = UniversalAnnotateApp()
    window.show()
    sys.exit(app.exec())