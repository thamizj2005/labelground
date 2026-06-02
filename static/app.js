const state = {
    currentProject: null,
    currentImage: null,
    currentAnnotations: [],
    annotationType: null,
    classes: [],
    zoom: 1,
    panOffset: { x: 0, y: 0 },
    isDrawing: false,
    currentTool: null,
    tempAnnotations: [],
    currentPolygon: null,
    currentKeypointIndex: 0,
    currentClassId: 0,
    needsInitialFit: false,
    isDirty: false,  // Track if annotations have been modified
    // Selection and editing state
    selectedAnnotation: null,
    isDragging: false,
    isResizing: false,
    resizeHandle: null,
    dragOffset: { x: 0, y: 0 },
    originalAnnotation: null,  // For undo on drag/resize
    // Polygon point editing
    isDraggingPoint: false,
    dragPointIndex: -1,
    hoveredEdgePoint: null, // For visual feedback when hovering polygon edges
    hoveredEdgeIndex: -1,
    autoAI: false, // Streaming AI mode
    activeBatchTaskId: null, // Track active batch annotation task
    showUnverifiedOnly: false, // Filter for verification mode
    activeTrainingTaskId: null, // Track active training task
    logs: [], // Application logs/notifications
    shuffledProjectId: null, // ID of project that is currently shuffled
    user: null, // Current logged in user
    token: localStorage.getItem('token') || null, // JWT token
    isConfusionSorted: false // Active Learning sorting
};

// Global variables for canvas interaction
let isPanning = false;
let lastPanPos = { x: 0, y: 0 };

// Canvas setup
const canvas = document.getElementById('annotation-canvas');
const ctx = canvas.getContext('2d');
let imageObj = null;

// API base URL
const API_BASE = window.location.origin + '/api';

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    // 10 Min Auto-Refresh (forces re-login because token is cleared)
    setTimeout(() => {
        window.location.reload();
    }, 10 * 60 * 1000);

    // Force login on every refresh by clearing any persisted token
    localStorage.removeItem('token');
    state.token = null;
    state.user = null;
    updateUserUI();
    showLandingPage();

    setupEventListeners();
    setupCanvas();

    // Resume training progress tracking if one was active
    const savedTaskId = localStorage.getItem('activeTrainingTaskId');
    if (savedTaskId) {
        state.activeTrainingTaskId = savedTaskId;
        pollTrainingProgress(savedTaskId);
    }

    // Landing Page Enter Key Listener
    document.addEventListener('keydown', (e) => {
        const landing = document.getElementById('landing-page');
        if (landing && landing.style.display !== 'none' && e.key === 'Enter') {
            const authModal = document.getElementById('auth-modal');
            if (!authModal || authModal.style.display === 'none') {
                enterApp();
            }
        }
    });
});

function enterApp() {
    if (!state.user) {
        showAuthModal('login');
        return;
    }
    const landing = document.getElementById('landing-page');
    if (landing) {
        landing.style.opacity = '0';
        setTimeout(() => {
            landing.style.display = 'none';
            showDashboard();
        }, 500);
    }
}

function showLandingPage() {
    const landing = document.getElementById('landing-page');
    if (landing) {
        landing.style.display = 'flex';
        landing.style.opacity = '1';
    }
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (sidebar) {
        sidebar.classList.toggle('active');
        // Give it a moment for the transition before resizing
        setTimeout(resizeCanvas, 300);
    }
}

function toggleAnnotationsPanel() {
    const panel = document.getElementById('instance-panel');
    const reopenBtn = document.getElementById('annotations-reopen-btn');
    const toggleBtn = document.getElementById('annotations-toggle-btn');
    if (!panel) return;

    const isCollapsed = panel.classList.contains('panel-collapsed');
    if (isCollapsed) {
        // Show panel
        panel.classList.remove('panel-collapsed');
        if (reopenBtn) reopenBtn.style.display = 'none';
        if (toggleBtn) { toggleBtn.textContent = '◀'; toggleBtn.title = 'Hide Annotations Panel'; }
    } else {
        // Hide panel
        panel.classList.add('panel-collapsed');
        if (reopenBtn) reopenBtn.style.display = 'block';
        if (toggleBtn) { toggleBtn.textContent = '▶'; toggleBtn.title = 'Show Annotations Panel'; }
    }
    // Resize canvas to fill newly available space
    setTimeout(resizeCanvas, 320);
}

function showDashboard() {
    document.getElementById('dashboard-view').style.display = 'flex';
    document.getElementById('workspace-view').style.display = 'none';
    const manageView = document.getElementById('manage-projects-view');
    if (manageView) manageView.style.display = 'none';
    const exportView = document.getElementById('export-project-view');
    if (exportView) exportView.style.display = 'none';
    const analyticsView = document.getElementById('analytics-view');
    if (analyticsView) analyticsView.style.display = 'none';
    const logsView = document.getElementById('logs-view');
    if (logsView) logsView.style.display = 'none';
    const userView = document.getElementById('user-management-view');
    if (userView) userView.style.display = 'none';
}

/**
 * Resets the application state to a clean slate.
 * Used when a project is deleted or before loading a new project.
 */
function clearState() {
    state.currentProject = null;
    state.currentImage = null;
    state.currentAnnotations = [];
    state.tempAnnotations = [];
    state.annotationType = null;
    state.classes = [];
    state.isDirty = false;
    state.images = [];
    state.totalImages = 0;

    // Reset canvas and image object
    imageObj = null;
    if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    // Reset history
    state.history = [];
    state.historyIndex = -1;

    // Reset interaction state
    state.selectedAnnotation = null;
    state.isDragging = false;
    state.isResizing = false;
    state.currentPolygon = null;

    // Update UI elements if they exist
    const imageList = document.getElementById('image-list');
    if (imageList) imageList.innerHTML = '';

    const counterElem = document.getElementById('image-counter');
    if (counterElem) counterElem.textContent = '';

    const projectInfoElem = document.getElementById('project-info-minimal');
    if (projectInfoElem) projectInfoElem.textContent = '';
}

function showWorkspace() {
    document.getElementById('dashboard-view').style.display = 'none';
    document.getElementById('manage-projects-view').style.display = 'none';
    document.getElementById('workspace-view').style.display = 'block';
    // Hide sidebar on mobile by default when entering workspace
    if (window.innerWidth <= 768) {
        document.getElementById('sidebar').classList.remove('active');
    }
    resizeCanvas();
}

function showManageProjectsView() {
    document.getElementById('dashboard-view').style.display = 'none';
    document.getElementById('workspace-view').style.display = 'none';
    document.getElementById('manage-projects-view').style.display = 'flex';
    const exportView = document.getElementById('export-project-view');
    if (exportView) exportView.style.display = 'none';
    loadManageProjects();
}


function showHelpView() {
    const modal = document.getElementById('help-modal');
    if (modal) {
        modal.style.display = 'flex';
        switchHelpTab('shortcuts'); // Default to shortcuts
    }
}

function hideHelpModal() {
    document.getElementById('help-modal').style.display = 'none';
}

function switchHelpTab(tabName) {
    // Hide all sections
    ['shortcuts', 'ai', 'workflow', 'about'].forEach(tab => {
        const section = document.getElementById(`help-section-${tab}`);
        if (section) section.style.display = 'none';

        // Reset link style
        const link = document.getElementById(`help-tab-${tab}`);
        if (link) {
            link.style.background = 'transparent';
            link.style.fontWeight = 'normal';
            link.style.color = '#666';
        }
    });

    // Show selected section
    const activeSection = document.getElementById(`help-section-${tabName}`);
    if (activeSection) activeSection.style.display = 'block';

    // Highlight selected link
    const activeLink = document.getElementById(`help-tab-${tabName}`);
    if (activeLink) {
        activeLink.style.background = '#e9ecef';
        activeLink.style.fontWeight = 'bold';
        activeLink.style.color = '#2c3e50';
    }
}

async function loadManageProjects() {
    try {
        const projects = await apiRequest('/projects');
        const grid = document.getElementById('manage-projects-grid');
        grid.innerHTML = '';

        for (const project of projects) {
            // Create card
            const card = document.createElement('div');
            card.className = 'dashboard-card';
            card.style.alignItems = 'flex-start';
            card.style.textAlign = 'left';
            card.style.position = 'relative';
            card.style.cursor = 'default'; // Don't use pointer for whole card since it has buttons

            // Fetch an image for thumbnail (optional, doing simple placeholder for now or first image)
            // Ideally we'd have a project thumbnail field, but we can fetch one image
            // Fetch first image for thumbnail
            let thumbnailHtml = `<div style="width: 100%; height: 150px; background: #f0f2f5; border-radius: 8px 8px 0 0; margin-bottom: 0; display: flex; align-items: center; justify-content: center; overflow: hidden; border-bottom: 1px solid #eee;">
                <span style="font-size: 3rem;">📁</span>
            </div>`;

            if (project.image_count > 0) {
                // Fetch the latest image for thumbnail (using limit=1 and assuming backend sorts or we can sort)
                try {
                    // Add timestamp to project images request to avoid caching
                    const timestamp = new Date().getTime();
                    const imagesData = await apiRequest(`/projects/${project.id}/images?limit=1&_t=${timestamp}`);
                    if (imagesData.images && imagesData.images.length > 0) {
                        const img = imagesData.images[0];
                        thumbnailHtml = `<div style="width: 100%; height: 150px; background: #fff; border-radius: 8px 8px 0 0; margin-bottom: 0; overflow: hidden; border-bottom: 1px solid #eee;">
                            <img src="${API_BASE}/images/${img.id}/file?_t=${timestamp}" style="width: 100%; height: 100%; object-fit: cover;" alt="${project.name}">
                        </div>`;
                    }
                } catch (e) {
                    console.error('Failed to load thumbnail', e);
                }
            }

            card.innerHTML = `
                ${thumbnailHtml}
                <div style="padding: 15px; width: 100%;">
                    <h3 style="margin: 0 0 5px 0; font-size: 1.1rem; color: #2c3e50;">${project.name}</h3>
                    <p style="margin: 0 0 5px 0; font-size: 0.9rem; color: #7f8c8d;">${project.annotation_type} &bull; ${project.image_count || 0} images</p>
                    <p style="margin: 0 0 12px 0; font-size: 0.75rem; color: #bdc3c7;">
                        ${project.role === 'owner' ? '👑 Owner' : '👤 Member'}
                    </p>
                    <div style="display: flex; gap: 8px;">
                        <button class="btn btn-primary" style="flex: 2; padding: 9px; font-weight: bold;" onclick="event.stopPropagation(); loadProject(${project.id})">Open</button>
                        <button class="btn btn-secondary" style="flex: 1; padding: 8px; font-size: 0.85rem;" title="Manage Team" onclick="event.stopPropagation(); showTeamModal(${project.id}, '${project.name}', '${project.role}')">👥</button>
                        <button class="btn btn-secondary" style="flex: 1; padding: 8px; font-size: 0.85rem;" title="Project Analytics" onclick="event.stopPropagation(); showProjectAnalyticsModal(${project.id})">📊</button>
                        <button class="btn btn-danger" style="flex: 1; padding: 8px; font-size: 0.85rem;" onclick="event.stopPropagation(); deleteProjectWithConfirmation(${project.id}, '${project.name}')">🗑</button>
                    </div>
                </div>
            `;

            // Adjust card padding to accommodate new layout
            card.style.padding = '0';
            card.style.overflow = 'hidden';

            grid.appendChild(card);

        }
    } catch (error) {
        console.error('Failed to load manage projects:', error);
    }
}

async function deleteProjectWithConfirmation(projectId, projectName) {
    const confirmation = prompt(`Are you sure you want to delete project "${projectName}"? This action cannot be undone.\n\nType "confirm" to proceed.`);


    if (confirmation === 'confirm') {
        try {
            await apiRequest(`/projects/${projectId}`, { method: 'DELETE' });

            // If the deleted project was the active one, clear state and go home
            if (state.currentProject && state.currentProject.id === projectId) {
                clearState();
                showDashboard();
            }

            showToast('Project deleted successfully.', 'success');
            loadManageProjects(); // Reload grid
        } catch (error) {
            showToast('Failed to delete project: ' + error.message, 'error');
        }
    } else {
        if (confirmation !== null) {
            showToast('Deletion cancelled. You must type "confirm" exactly.', 'warning');
        }
    }
}

// ─── Team Management ──────────────────────────────────────────────────────────

let _teamModalProjectId = null;

async function showTeamModal(projectId, projectName, role) {
    _teamModalProjectId = projectId;

    // Build/show modal
    let modal = document.getElementById('team-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'team-modal';
        modal.style.cssText = 'display:none;position:fixed;inset:0;background:rgba(0,0,0,0.5);z-index:9000;align-items:center;justify-content:center;';
        modal.innerHTML = `
            <div style="background:#fff;border-radius:16px;width:520px;max-width:95vw;max-height:85vh;overflow:auto;box-shadow:0 20px 60px rgba(0,0,0,0.3);">
                <div style="padding:24px 28px 16px;border-bottom:1px solid #eee;display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <h2 id="team-modal-title" style="margin:0;font-size:1.3rem;color:#2c3e50;">👥 Team</h2>
                        <p id="team-modal-sub" style="margin:4px 0 0;font-size:13px;color:#aaa;"></p>
                    </div>
                    <button onclick="document.getElementById('team-modal').style.display='none'" style="background:none;border:none;font-size:22px;cursor:pointer;color:#999;">✕</button>
                </div>
                <div id="team-members-list" style="padding:20px 28px;min-height:100px;"></div>
                <div id="team-add-section" style="padding:0 28px 24px;">
                    <hr style="margin:0 0 16px;border-color:#f0f0f0;">
                    <p style="font-size:12px;font-weight:700;color:#aaa;margin:0 0 10px;letter-spacing:0.5px;">ADD MEMBER</p>
                    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
                        <input id="team-add-username" class="form-control" placeholder="Username" style="flex:2;min-width:140px;">
                        <select id="team-add-canedit" class="form-control" style="flex:1;min-width:110px;">
                            <option value="true">Can Edit</option>
                            <option value="false">View Only</option>
                        </select>
                        <button class="btn btn-primary" onclick="addProjectMember()" style="padding:9px 16px;">Add</button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    modal.style.display = 'flex';
    document.getElementById('team-modal-title').textContent = `👥 ${projectName}`;
    document.getElementById('team-modal-sub').textContent = 'Team members who have access to this project';
    document.getElementById('team-add-section').style.display = (role === 'owner' || (state.user && state.user.role === 'admin')) ? 'block' : 'none';

    await refreshTeamMembersList(projectId, role);
}

async function refreshTeamMembersList(projectId, role) {
    const list = document.getElementById('team-members-list');
    list.innerHTML = '<div style="color:#aaa;text-align:center;padding:20px;">Loading...</div>';
    try {
        const members = await apiRequest(`/projects/${projectId}/members`);
        if (!members.length) {
            list.innerHTML = '<div style="color:#aaa;text-align:center;padding:20px;">No members yet</div>';
            return;
        }
        list.innerHTML = members.map(m => `
            <div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid #f5f5f5;">
                <div style="width:36px;height:36px;background:${m.is_owner ? '#e74c3c' : '#3498db'};color:white;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:bold;flex-shrink:0;">
                    ${m.username[0].toUpperCase()}
                </div>
                <div style="flex:1;">
                    <div style="font-weight:600;color:#2c3e50;">${m.username}</div>
                    <div style="font-size:12px;color:#aaa;">
                        ${m.is_owner ? '👑 Owner' : (m.can_edit ? '✏️ Can Edit' : '👁 View Only')} &nbsp;•&nbsp; ${m.role}
                    </div>
                </div>
                ${!m.is_owner && (role === 'owner' || (state.user && state.user.role === 'admin')) ? `
                    <button onclick="removeProjectMember(${projectId}, '${m.username}', '${role}')" 
                        style="background:none;border:none;color:#e74c3c;cursor:pointer;font-size:13px;padding:4px 8px;border-radius:4px;opacity:0.7;"
                        title="Remove member">✕ Remove</button>
                ` : ''}
            </div>
        `).join('');
    } catch (err) {
        list.innerHTML = `<div style="color:#e74c3c;text-align:center;padding:20px;">Failed to load members: ${err.message}</div>`;
    }
}

async function addProjectMember() {
    const username = document.getElementById('team-add-username').value.trim();
    const canEdit = document.getElementById('team-add-canedit').value === 'true';
    if (!username) { showToast('Please enter a username', 'warning'); return; }
    try {
        await apiRequest(`/projects/${_teamModalProjectId}/members`, {
            method: 'POST',
            body: JSON.stringify({ username, can_edit: canEdit, can_export: false })
        });
        showToast(`✅ ${username} added to project`, 'success');
        document.getElementById('team-add-username').value = '';
        await refreshTeamMembersList(_teamModalProjectId, 'owner');
    } catch (err) {
        showToast('❌ ' + (err.message || 'Failed to add member'), 'error');
    }
}

async function removeProjectMember(projectId, username, role) {
    if (!confirm(`Remove "${username}" from this project?`)) return;
    try {
        await apiRequest(`/projects/${projectId}/members/${username}`, { method: 'DELETE' });
        showToast(`✅ ${username} removed`, 'success');
        await refreshTeamMembersList(projectId, role);
    } catch (err) {
        showToast('❌ ' + (err.message || 'Failed to remove member'), 'error');
    }
}

// API Functions

async function apiRequest(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': state.token ? `Bearer ${state.token}` : '',
                ...options.headers
            },
            ...options
        });

        if (!response.ok) {
            if (response.status === 401) {
                // Token likely expired or invalid
                console.warn('Unauthorized request, clearing token and showing login.');
                state.token = null;
                state.user = null;
                localStorage.removeItem('token');
                if (typeof showAuthModal === 'function') {
                    showAuthModal('login');
                }
                throw new Error('Session expired. Please log in again.');
            }
            const error = await response.json();
            throw new Error(error.detail || `API Error: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API Request failed:', error);
        if (typeof showToast === 'function' && !error.message.includes('Session expired')) {
            showToast(`Error: ${error.message}`, 'error');
        } else if (typeof showToast !== 'function') {
            console.error(`Error: ${error.message}`);
        }
        throw error;
    }
}

// User Management Functions
async function loginUser(username, password) {
    try {
        const formData = new FormData();
        formData.append('username', username);
        formData.append('password', password);

        const response = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Login failed');
        }

        const data = await response.json();
        state.token = data.access_token;
        localStorage.setItem('token', data.access_token);

        // Get user profile
        const user = await apiRequest('/auth/me');
        state.user = user;

        updateUserUI();
        hideAuthModal();
        enterApp();
        loadProjects();
        showToast(`Welcome back, ${user.username}!`, 'success');
    } catch (error) {
        showToast(error.message, 'error');
    }
}

async function registerUser(username, password) {
    try {
        await apiRequest('/auth/register', {
            method: 'POST',
            body: JSON.stringify({ username, password })
        });
        showToast('Registration successful! Please login.', 'success');
        showAuthModal('login');
    } catch (error) {
        showToast(error.message, 'error');
    }
}

function logoutUser() {
    state.user = null;
    state.token = null;
    localStorage.removeItem('token');
    clearState();
    updateUserUI();
    showLandingPage();
    showToast('Logged out successfully', 'info');
}

function updateUserUI() {
    const userDisplay = document.getElementById('user-display');
    if (userDisplay) {
        if (state.user) {
            userDisplay.innerHTML = `
                <div style="display: flex; align-items: center; gap: 8px; position: relative;" id="user-avatar-container">
                    <div onclick="toggleUserMenu()" style="display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 4px 8px; border-radius: 20px; transition: background 0.2s;" onmouseover="this.style.background='rgba(255,255,255,0.15)'" onmouseout="this.style.background='transparent'">
                        <div style="width: 30px; height: 30px; background: linear-gradient(135deg, #3498db, #2980b9); color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 13px; flex-shrink: 0;">
                            ${state.user.username[0].toUpperCase()}
                        </div>
                        <div style="text-align: left;">
                            <div style="font-weight: 600; font-size: 13px; color: white; line-height: 1.1;">${state.user.username}</div>
                            <div style="font-size: 9px; color: rgba(255,255,255,0.55); text-transform: uppercase; letter-spacing: 0.5px;">${state.user.role}</div>
                        </div>
                        <span style="color: rgba(255,255,255,0.5); font-size: 10px;">▾</span>
                    </div>
                    <div id="user-dropdown-menu" style="display: none; position: absolute; top: calc(100% + 8px); right: 0; background: white; border-radius: 10px; box-shadow: 0 8px 24px rgba(0,0,0,0.18); min-width: 180px; z-index: 9999; overflow: hidden; border: 1px solid #eee;">
                        <div style="padding: 12px 16px; border-bottom: 1px solid #f0f0f0; background: #f8f9fa;">
                            <div style="font-weight: 700; font-size: 13px; color: #2c3e50;">${state.user.username}</div>
                            <div style="font-size: 11px; color: #7f8c8d; text-transform: capitalize;">${state.user.role}</div>
                        </div>
                        <button onclick="showChangePasswordModal(); toggleUserMenu()" style="width: 100%; text-align: left; padding: 10px 16px; background: none; border: none; cursor: pointer; font-size: 13px; color: #2c3e50; display: flex; align-items: center; gap: 8px;" onmouseover="this.style.background='#f8f9fa'" onmouseout="this.style.background='none'">🔑 Change Password</button>
                        <button onclick="showDashboard()" style="width: 100%; text-align: left; padding: 10px 16px; background: none; border: none; cursor: pointer; font-size: 13px; color: #2c3e50; display: flex; align-items: center; gap: 8px;" onmouseover="this.style.background='#f8f9fa'" onmouseout="this.style.background='none'">🏠 Dashboard</button>
                        <div style="border-top: 1px solid #f0f0f0;"></div>
                        <button onclick="logoutUser()" style="width: 100%; text-align: left; padding: 10px 16px; background: none; border: none; cursor: pointer; font-size: 13px; color: #e74c3c; display: flex; align-items: center; gap: 8px;" onmouseover="this.style.background='#fff5f5'" onmouseout="this.style.background='none'">🚪 Logout</button>
                    </div>
                </div>
            `;
            const userCard = document.getElementById('user-management-card');
            if (userCard) {
                if (state.user.role === 'admin' || state.user.role === 'sub_admin') {
                    userCard.style.display = 'block';
                } else {
                    userCard.style.display = 'none';
                }
            }

            const exportCard = document.getElementById('export-project-card');
            if (exportCard) {
                // Main admin and sub admin can export. Annotators cannot.
                if (state.user.role === 'admin' || state.user.role === 'sub_admin') {
                    exportCard.style.display = 'block';
                } else {
                    exportCard.style.display = 'none';
                }
            }

            const createProjectCard = document.getElementById('create-project-card');
            if (createProjectCard) {
                if (state.user.role === 'admin' || state.user.role === 'sub_admin') {
                    createProjectCard.style.display = 'block';
                } else {
                    createProjectCard.style.display = 'none';
                }
            }
        } else {
            userDisplay.innerHTML = '';
        }
    }
}

function toggleUserMenu() {
    const menu = document.getElementById('user-dropdown-menu');
    if (!menu) return;
    const isVisible = menu.style.display === 'block';
    menu.style.display = isVisible ? 'none' : 'block';
}

// Close user dropdown when clicking outside
document.addEventListener('click', function (e) {
    const container = document.getElementById('user-avatar-container');
    if (container && !container.contains(e.target)) {
        const menu = document.getElementById('user-dropdown-menu');
        if (menu) menu.style.display = 'none';
    }
});

function showAuthModal(mode = 'login') {
    const modal = document.getElementById('auth-modal');
    if (!modal) return;

    modal.style.display = 'flex';
    const loginForm = document.getElementById('login-form-container');
    const registerForm = document.getElementById('register-form-container');
    const changePwForm = document.getElementById('change-password-form-container');
    const forgotPwForm = document.getElementById('forgot-password-form-container');

    loginForm.style.display = 'none';
    registerForm.style.display = 'none';
    if (changePwForm) changePwForm.style.display = 'none';
    if (forgotPwForm) forgotPwForm.style.display = 'none';

    if (mode === 'login') {
        loginForm.style.display = 'block';
        document.getElementById('auth-modal-title').textContent = 'Login to Labelground';
    } else if (mode === 'register') {
        registerForm.style.display = 'block';
        document.getElementById('auth-modal-title').textContent = 'Create Account';
    } else if (mode === 'changePassword') {
        const title = document.getElementById('auth-modal-title');
        if (title) title.textContent = 'Change Password';
        if (changePwForm) changePwForm.style.display = 'block';
    } else if (mode === 'forgotPassword') {
        const title = document.getElementById('auth-modal-title');
        if (title) title.textContent = 'Forgot Password';
        if (forgotPwForm) {
            forgotPwForm.style.display = 'block';
            document.getElementById('forgot-password-step-1').style.display = 'block';
            document.getElementById('forgot-password-step-2').style.display = 'none';
            document.getElementById('forgot-pw-username').value = '';
        }
    }
}

// ─── UI Helpers ──────────────────────────────────────────────────────────────
window.togglePasswordVis = function (btn) {
    const input = btn.previousElementSibling;
    if (input && input.tagName === 'INPUT') {
        if (input.type === 'password') {
            input.type = 'text';
            btn.textContent = '🙈';
        } else {
            input.type = 'password';
            btn.textContent = '👁️';
        }
    }
};

// ─── Security Question Flow ───────────────────────────────────────────────────
async function fetchSecurityQuestion() {
    const username = document.getElementById('forgot-pw-username').value.trim();
    if (!username) {
        showToast('Please enter your username', 'warning');
        return;
    }
    try {
        const res = await apiRequest(`/auth/security-question?username=${encodeURIComponent(username)}`, { method: 'GET' });
        if (res.question) {
            document.getElementById('forgot-pw-question-text').textContent = res.question;
            document.getElementById('forgot-password-step-1').style.display = 'none';
            document.getElementById('forgot-password-step-2').style.display = 'block';
        }
    } catch (e) {
        showToast(e.message, 'error');
    }
}

async function submitSecurityReset() {
    const username = document.getElementById('forgot-pw-username').value.trim();
    const answer = document.getElementById('forgot-pw-answer').value.trim();
    const newPass = document.getElementById('forgot-pw-new').value.trim();

    if (!answer || !newPass) {
        showToast('Please fill all fields', 'warning');
        return;
    }

    try {
        await apiRequest('/auth/reset-password-security', {
            method: 'POST',
            body: JSON.stringify({
                username: username,
                answer: answer,
                new_password: newPass
            })
        });
        showToast('Password reset successfully! Please log in.', 'success');
        showAuthModal('login');
    } catch (e) {
        showToast(e.message, 'error');
    }
}

// ─── User Management ──────────────────────────────────────────────────────────

function showUserManagementView() {
    document.getElementById('dashboard-view').style.display = 'none';
    const el = document.getElementById('user-management-view');
    if (el) el.style.display = 'flex';
    loadUsersList();
}

async function loadUsersList() {
    try {
        const users = await apiRequest('/users');
        const tbody = document.getElementById('user-management-body');
        if (!tbody) return;
        tbody.innerHTML = '';
        let globalUsersData = {};
        users.forEach(u => {
            globalUsersData[u.id] = u;
            const roleBadgeColor = u.role === 'admin' ? '#e74c3c' : (u.role === 'sub_admin' ? '#ff9f43' : '#3498db');

            // Sub-admins should not see an edit btn for main admins or themselves if we strictly isolate.
            // As a quick UX filter, show edit if (main admin AND not self-admin lock) OR (subadmin AND user is annotator)
            let canEdit = false;
            if (state.user && state.user.role === 'admin' && !(u.role === 'admin' && u.id !== state.user.id)) {
                canEdit = true;
            } else if (state.user && state.user.role === 'sub_admin' && u.role === 'annotator' && u.created_by_id === state.user.id) {
                canEdit = true;
            }

            const editBtnHtml = canEdit ? `<button class="btn btn-secondary" onclick="showEditUserModal(${u.id})" style="padding: 4px 10px; font-size: 0.8rem;">✏️ Edit</button>` : '';

            tbody.innerHTML += `
                <tr style="border-bottom: 1px solid #f0f3f4; transition: background 0.2s;">
                    <td style="padding: 15px 20px; color: #34495e;">#${u.id}</td>
                    <td style="padding: 15px 20px; color: #2c3e50; font-weight: 500;">${u.username}</td>
                    <td style="padding: 15px 20px;">
                        <span style="background: ${roleBadgeColor}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; text-transform: uppercase;">${u.role}</span>
                    </td>
                    <td style="padding: 15px 20px;">
                        ${editBtnHtml}
                    </td>
                </tr>
            `;
        });
        window._globalUsersData = globalUsersData;
    } catch (e) {
        showToast('Failed to load users: ' + e.message, 'error');
    }
}

function showCreateUserModal() {
    const modal = document.getElementById('create-user-modal');
    if (modal) {
        modal.style.display = 'flex';
        const roleSelect = document.getElementById('create-user-role-select');
        roleSelect.innerHTML = '<option value="annotator">Annotator</option>';
        if (state.user && state.user.role === 'admin') {
            roleSelect.innerHTML += '<option value="sub_admin">Sub-Administrator</option>';
        }
    }
}

async function submitCreateUser(form) {
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    try {
        await apiRequest('/users', {
            method: 'POST',
            body: JSON.stringify(data)
        });
        document.getElementById('create-user-modal').style.display = 'none';
        showToast('User created successfully', 'success');
        form.reset();
        loadUsersList();
    } catch (e) {
        showToast('Failed to create user: ' + e.message, 'error');
    }
}

window.showEditUserModal = function (userId, usersData) {
    const u = (usersData && usersData[userId]) || window._globalUsersData[userId];
    if (!u) return;

    const modal = document.getElementById('edit-user-modal');
    if (modal) {
        document.getElementById('edit-user-id').value = u.id;
        document.getElementById('edit-user-username').value = u.username;
        document.getElementById('edit-user-form').password.value = ''; // clear password field
        const roleSelect = document.getElementById('edit-user-role-select');

        roleSelect.innerHTML = '<option value="annotator">Annotator</option>';
        if (state.user && state.user.role === 'admin') {
            roleSelect.innerHTML += '<option value="sub_admin">Sub-Administrator</option>';
            roleSelect.innerHTML += '<option value="admin">Administrator</option>';
        }
        roleSelect.value = u.role;
        modal.style.display = 'flex';
    }
};

window.submitEditUser = async function (form) {
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    const userId = data.user_id;

    // Cleanup empty strings
    Object.keys(data).forEach(k => {
        if (data[k] === '') delete data[k];
    });
    delete data.user_id;

    try {
        await apiRequest(`/users/${userId}`, {
            method: 'PATCH',
            body: JSON.stringify(data)
        });
        document.getElementById('edit-user-modal').style.display = 'none';
        showToast('User updated successfully', 'success');
        loadUsersList();
    } catch (e) {
        showToast('Failed to update user: ' + e.message, 'error');
    }
}

function showChangePasswordModal() {
    showAuthModal('changePassword');
}

async function changeUserPassword() {
    const oldPass = document.getElementById('change-pw-old').value.trim();
    const newPass = document.getElementById('change-pw-new').value.trim();
    const confirmPass = document.getElementById('change-pw-confirm').value.trim();

    if (!oldPass || !newPass || !confirmPass) {
        showToast('Please fill all fields', 'warning');
        return;
    }
    if (newPass !== confirmPass) {
        showToast('New passwords do not match', 'error');
        return;
    }
    if (newPass.length < 6) {
        showToast('Password must be at least 6 characters', 'warning');
        return;
    }

    try {
        await apiRequest('/auth/change-password', {
            method: 'POST',
            body: JSON.stringify({ old_password: oldPass, new_password: newPass })
        });
        showToast('✅ Password changed successfully!', 'success');
        hideAuthModal();
    } catch (err) {
        showToast('❌ ' + (err.message || 'Failed to change password'), 'error');
    }
}

async function adminResetPassword() {
    const username = document.getElementById('admin-reset-username').value.trim();
    const newPass = document.getElementById('admin-reset-newpass').value.trim();
    if (!username || !newPass) {
        showToast('Please fill all fields', 'warning');
        return;
    }
    if (newPass.length < 6) {
        showToast('Password must be at least 6 characters', 'warning');
        return;
    }
    try {
        await apiRequest('/auth/admin-reset-password', {
            method: 'POST',
            body: JSON.stringify({ username, new_password: newPass })
        });
        showToast(`✅ Password for "${username}" reset successfully!`, 'success');
    } catch (err) {
        showToast('❌ ' + (err.message || 'Reset failed'), 'error');
    }
}

function hideAuthModal() {
    const modal = document.getElementById('auth-modal');
    if (modal) modal.style.display = 'none';
}


// Project Management
async function loadProjects() {
    try {
        const projects = await apiRequest('/projects');
        const projectList = document.getElementById('project-list');
        projectList.innerHTML = '';

        projects.forEach(project => {
            const div = document.createElement('div');
            div.className = 'project-card';
            if (state.currentProject && state.currentProject.id === project.id) {
                div.classList.add('active');
            }

            div.innerHTML = `
                <strong>${project.name}</strong><br>
                <small>${project.annotation_type} • ${project.image_count || 0} images</small>
            `;

            div.onclick = () => loadProject(project.id);
            projectList.appendChild(div);
        });
    } catch (error) {
        console.error('Failed to load projects:', error);
    }
}

async function loadProject(projectId) {
    try {
        // Clear previous project state before loading new one
        clearState();

        // Load project details
        const project = await apiRequest(`/projects/${projectId}`);

        if (!project) {
            showToast('Project data is empty', 'error');
            return;
        }

        state.currentProject = project;
        state.annotationType = project.annotation_type || 'bbox';
        state.classes = (project.label_schema && project.label_schema.classes) ? project.label_schema.classes : [];

        // Load auto-train settings
        const autoTrainEnabled = localStorage.getItem(`project_${projectId}_auto_train`) !== 'false';
        const autoFinetuneEnabled = localStorage.getItem(`project_${projectId}_auto_finetune`) !== 'false';

        const autoTrainCheckbox = document.getElementById('auto-train-checkbox');
        if (autoTrainCheckbox) {
            autoTrainCheckbox.checked = autoTrainEnabled;
        }

        const autoFinetuneCheckbox = document.getElementById('auto-finetune-checkbox');
        if (autoFinetuneCheckbox) {
            autoFinetuneCheckbox.checked = autoFinetuneEnabled;
        }

        // Update UI (Safely check if elements exist after UI cleanup)
        const titleElem = document.getElementById('project-title');
        if (titleElem) titleElem.textContent = project.name;

        const infoElem = document.getElementById('project-info');
        if (infoElem) {
            infoElem.innerHTML = `Type: ${project.annotation_type} • Created: ${new Date(project.created_at).toLocaleDateString()}`;
        }

        // Update project list highlighting
        document.querySelectorAll('.project-card').forEach(card => {
            card.classList.remove('active');
        });

        // Load project images
        await loadProjectImages(projectId);

        // Update toolbar based on annotation type
        updateToolbar();

        // Auto-select first tool
        if (state.annotationType === 'bbox') setTool('bbox');
        else if (state.annotationType === 'polygon') setTool('auto-segment');
        else if (state.annotationType === 'keypoints') setTool('keypoints');

        // Load global actions
        updateGlobalActions();

        // Reload projects to update highlighting
        // Reload projects to update highlighting
        loadProjects();

        // Switch to workspace view
        showWorkspace();

    } catch (error) {
        console.error('Failed to load project:', error);
        showToast('Error loading project: ' + error.message, 'error');
    }
}

async function createProject(projectData) {
    try {
        await apiRequest('/projects', {
            method: 'POST',
            body: JSON.stringify(projectData)
        });

        hideCreateProjectModal();
        loadProjects();
        alert('Project created successfully! Go to "Manage Projects" to open it.');

        // Auto-refresh manage projects view if open
        if (document.getElementById('manage-projects-view').style.display !== 'none') {
            loadManageProjects();
        }

    } catch (error) {
        alert('Failed to create project');
    }
}

// Image Management
async function loadProjectImages(projectId, skip = 0, limit = 1000) {
    const imageList = document.getElementById('image-list');
    if (!imageList) return;

    try {
        const timestamp = new Date().getTime();
        const sortParam = state.isConfusionSorted ? 'confidence' : 'id';
        const data = await apiRequest(`/projects/${projectId}/images?skip=${skip}&limit=${limit}&sort=${sortParam}&_t=${timestamp}`);

        // Save to state for navigation and deletion
        state.images = data.images;
        state.totalImages = data.total;

        // Check if shuffle is enabled (stored in localStorage)
        const key = `project_${projectId}_shuffle_images`;
        const shuffleEnabled = localStorage.getItem(key) === 'true';

        // Update button appearance
        const shuffleBtn = document.querySelector('button[onclick="toggleImageShuffle()"]');
        if (shuffleEnabled) {
            state.images = shuffleArray(state.images);
            state.shuffledProjectId = projectId;
        } else {
            state.shuffledProjectId = null;
        }

        let unverifiedHtml = '';
        let verifiedHtml = '';

        let unverifiedCount = 0;
        let verifiedCount = 0;

        state.images.forEach(image => {
            const isVerified = image.verification_status === 'verified';

            let statusHtml = '';
            if (image.verification_status === 'needs_edit') {
                statusHtml = `<span class="status-badge" style="background: #e74c3c; color: white;">Edit Reset</span>`;
            } else if (isVerified) {
                statusHtml = `<span class="status-badge status-processed">✓</span>`;
            } else if (image.has_annotations && image.latest_created_by === 'auto') {
                statusHtml = `<span class="status-badge" style="background: #ff9f43; color: white;">AI (🟡)</span>`;
            } else if (!image.has_annotations) {
                statusHtml = `<span class="status-badge status-pending">New</span>`;
            } else {
                statusHtml = `<span class="status-badge status-processed">✓</span>`; // Fallback
            }

            const confidenceHtml = image.confidence < 1.0 ?
                `<span class="status-badge" style="background: ${image.confidence < 0.5 ? '#e74c3c' : '#f39c12'}; color: white;" title="AI Confidence: ${Math.round(image.confidence * 100)}%">${Math.round(image.confidence * 100)}%</span>` : '';

            const cardHtml = `
                <div class="project-card" id="image-card-${image.id}" style="padding: 10px; margin-bottom: 5px; cursor: pointer;" onclick="loadImage(${image.id})">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="font-size: 12px; display: block; overflow: hidden; text-overflow: ellipsis; max-width: 120px; white-space: nowrap;" title="${image.filename}">${image.filename}</strong>
                            <small>${image.dimensions.width}×${image.dimensions.height}</small>
                        </div>
                        <div style="display: flex; gap: 5px; align-items: center;">
                            ${confidenceHtml}
                            ${statusHtml}
                        </div>
                    </div>
                </div>
            `;

            if (isVerified) {
                verifiedHtml += cardHtml;
                verifiedCount++;
            } else {
                unverifiedHtml += cardHtml;
                unverifiedCount++;
            }
        });

        if (unverifiedCount === 0 && verifiedCount === 0) {
            imageList.innerHTML = '<p style="color: #aaa; padding: 10px;">No images yet</p>';
        } else {
            imageList.innerHTML = `
                ${unverifiedCount > 0 ? `<div style="padding: 5px 10px; font-weight: bold; font-size: 0.8rem; color: #7f8c8d; background: #fdfefe; position: sticky; top: 0; z-index: 5;">Unverified (${unverifiedCount})</div>${unverifiedHtml}` : ''}
                ${verifiedCount > 0 ? `
                    <div style="padding: 5px 10px; font-weight: bold; font-size: 0.8rem; color: #7f8c8d; background: #fdfefe; position: sticky; top: ${unverifiedCount > 0 ? '0' : '0'}; z-index: 5; margin-top: 10px; display: flex; justify-content: space-between; align-items: center; cursor: pointer;" onclick="const el=document.getElementById('verified-images-container'); el.style.display=el.style.display==='none'?'block':'none';">
                        <span>✅ Verified (${verifiedCount})</span>
                        <span style="font-size:0.7rem;">(Toggle)</span>
                    </div>
                    <div id="verified-images-container" style="display: none;">${verifiedHtml}</div>
                ` : ''}
            `;
        }
        updateCounter();

    } catch (error) {
        console.error('Failed to load images:', error);
        showToast('Error loading image list: ' + error.message, 'error');
    }
}

async function loadImage(imageId) {
    try {
        // Clear previous state
        state.currentImage = null;
        state.currentAnnotations = [];
        state.tempAnnotations = [];
        state.history = [];
        state.historyIndex = -1;

        // Clear in-progress drawing and selection states
        state.currentPolygon = null;
        state.selectedAnnotation = null;
        state.isDragging = false;
        state.isResizing = false;
        state.resizeHandle = null;

        // Load image
        const timestamp = new Date().getTime();
        const imageUrl = `${API_BASE}/images/${imageId}/file?_t=${timestamp}`;
        await loadImageToCanvas(imageUrl);

        // Load image metadata
        const images = await apiRequest(`/projects/${state.currentProject.id}/images`);
        const image = images.images.find(img => img.id === imageId);

        state.currentImage = { ...image, id: imageId };

        // Load all annotations into tempAnnotations
        await loadAllAnnotationVersions(imageId);

        // Update toolbar
        updateToolbar();

        // Update image list highlighting
        document.querySelectorAll('#image-list .project-card').forEach(card => {
            card.classList.remove('active');
        });
        const activeCard = document.getElementById(`image-card-${imageId}`);
        if (activeCard) activeCard.classList.add('active');

        updateCounter();

        // Draw annotations
        drawAnnotations();

        // Resume batch progress bar if active
        if (state.activeBatchTaskId) {
            pollBatchProgress(state.activeBatchTaskId);
        }

        // Streaming AI: If Auto-AI is on and image has no annotations, run AI automatically
        if (state.autoAI && state.tempAnnotations.length === 0) {
            console.log("🚀 Streaming AI: Automatically annotating...");
            runAutoAnnotate();
        }

    } catch (error) {
        console.error('Failed to load image:', error);
        showToast('Failed to load image: ' + error.message, 'error');
    }
}

function toggleConfusionSort() {
    state.isConfusionSorted = !state.isConfusionSorted;
    if (state.currentProject) {
        showToast(state.isConfusionSorted ? 'Prioritizing confused images (Active Learning)' : 'Standard sorting restored', 'info');
        loadProjectImages(state.currentProject.id);
    }
}

function updateCounter() {
    if (!state.images) return;

    // Find current index
    if (state.currentImage) {
        state.currentImageIndex = state.images.findIndex(img => img.id === state.currentImage.id);
    }

    const index = state.currentImageIndex;
    const total = state.totalImages || state.images.length;
    const counterText = index !== -1 ? `Image ${index + 1} / ${total}` : `${total} Images`;

    // Sidebar counter
    const counter = document.getElementById('image-counter');
    if (counter) counter.textContent = counterText;

    // Minimal top-bar info
    const minimalInfo = document.getElementById('project-info-minimal');
    if (minimalInfo && state.currentProject) {
        const type = state.currentProject.annotation_type === 'polygon' ? 'Seg' : 'Det';
        const shuffleStatus = state.shuffledProjectId === state.currentProject.id ? ' (🔀)' : '';
        minimalInfo.textContent = `${state.currentProject.name} | ${type} | ${counterText}${shuffleStatus}`;
    }
}

async function loadImageToCanvas(imageUrl) {
    return new Promise((resolve, reject) => {
        imageObj = new Image();
        // Removed crossOrigin = 'anonymous' to avoid CORS issues if server configuration is strict,
        // since we are fetching from the same origin anyway.
        imageObj.onload = () => {
            state.needsInitialFit = true;
            resetZoomAndFit();
            resolve();
        };
        imageObj.onerror = reject;
        imageObj.src = imageUrl;
    });
}

function resetZoomAndFit() {
    if (!imageObj) return;

    const container = canvas.parentElement;
    if (!container || container.clientWidth === 0 || container.clientHeight === 0) {
        console.warn('Canvas container has no size yet, retrying resetZoomAndFit...', {
            width: container ? container.clientWidth : 'no container',
            height: container ? container.clientHeight : 'no container'
        });
        // If layout isn't ready or hidden, try again in the next frame
        requestAnimationFrame(resetZoomAndFit);
        return;
    }
    console.log('Canvas container size:', container.clientWidth, 'x', container.clientHeight);

    // Set internal resolution to match displayed pixels
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    const padding = 20; // Slightly less padding for better view
    const availableWidth = Math.max(0, canvas.width - padding * 2);
    const availableHeight = Math.max(0, canvas.height - padding * 2);

    const scaleX = availableWidth / imageObj.width;
    const scaleY = availableHeight / imageObj.height;

    // Fit to view - Always fill the screen
    state.zoom = Math.min(scaleX, scaleY);

    // Center image
    state.panOffset.x = (canvas.width - imageObj.width * state.zoom) / 2;
    state.panOffset.y = (canvas.height - imageObj.height * state.zoom) / 2;

    state.needsInitialFit = false;
    updateZoomDisplay();
    drawImage();
}

async function importImages() {
    const imagePath = document.getElementById('image-path').value;
    const isFolder = document.getElementById('is-folder').checked;

    if (!imagePath) {
        alert('Please enter a path');
        return;
    }

    try {
        const result = await apiRequest(`/projects/${state.currentProject.id}/images`, {
            method: 'POST',
            body: JSON.stringify({
                paths: [imagePath],
                is_folder: isFolder
            })
        });

        hideImportModal();
        loadProjectImages(state.currentProject.id);
        alert(`Import completed! ${result.imported_count} images imported.`);

    } catch (error) {
        alert('Failed to import images');
    }
}

async function importVideo() {
    const videoPath = document.getElementById('video-path').value;
    const targetFps = parseFloat(document.getElementById('target-fps').value);

    if (!videoPath || !targetFps) {
        alert('Please enter video path and FPS');
        return;
    }

    try {
        await apiRequest(`/projects/${state.currentProject.id}/videos`, {
            method: 'POST',
            body: JSON.stringify({
                video_path: videoPath,
                fps: targetFps
            })
        });

        hideImportModal();
        alert('Video extraction started in background! Refresh images in a moment.');

    } catch (error) {
        alert('Failed to import video');
    }
}

// Annotation Management
async function loadAllAnnotationVersions(imageId) {
    try {
        // Load only the LATEST version of annotations (not all versions)
        const annotations = await apiRequest(`/images/${imageId}/annotations/latest`);

        state.tempAnnotations = annotations.map(ann => {
            const denormalized = { ...ann };
            const type = ann.type || state.annotationType;

            if (type === 'bbox' || (type === 'polygon' && !ann.points)) {
                denormalized.x = (ann.x || 0) * imageObj.width;
                denormalized.y = (ann.y || 0) * imageObj.height;
                denormalized.width = (ann.width || 0) * imageObj.width;
                denormalized.height = (ann.height || 0) * imageObj.height;
                denormalized.type = 'bbox'; // Ensure it's marked correctly if fallback occurred
            } else if (type === 'polygon') {
                denormalized.points = ann.points.map(p => ({
                    x: p.x * imageObj.width,
                    y: p.y * imageObj.height
                }));
            } else if (type === 'keypoints') {
                denormalized.points = (ann.points || []).map(p => p ? {
                    x: p.x * imageObj.width,
                    y: p.y * imageObj.height,
                    visible: p.visible
                } : null);
            }

            return denormalized;
        });

        // Reset dirty flag since we just loaded fresh annotations
        state.isDirty = false;

        drawAnnotations();
    } catch (error) {
        console.error('Failed to load annotations:', error);
    }
}

// Main save function
async function saveAnnotations(notify = true) {
    if (!state.currentProject || !state.currentImage || !imageObj) {
        if (notify) showToast('No active project or image selected to save', 'warning');
        return false;
    }

    // Allow saving empty annotations if they were deleted or for initial verification
    if (state.tempAnnotations.length === 0 && !state.isDirty && state.currentImage.status === 'processed') {
        if (notify) showToast('Image already verified', 'info');
        return false;
    }

    // Convert image coordinates (0 to imageObj.width/height) to normalized (0-1)
    const normalizedAnnotations = state.tempAnnotations.map(ann => {
        const normalized = { ...ann };
        delete normalized.drawing;
        delete normalized.tempPoint;
        delete normalized.source_version;

        const type = normalized.type || state.annotationType;
        const clamp = (v) => Math.max(0, Math.min(1, v));

        if (type === 'bbox') {
            normalized.x = clamp(normalized.x / imageObj.width);
            normalized.y = clamp(normalized.y / imageObj.height);
            normalized.width = clamp(normalized.width / imageObj.width);
            normalized.height = clamp(normalized.height / imageObj.height);
        } else if (type === 'polygon') {
            normalized.points = normalized.points.map(p => ({
                x: clamp(p.x / imageObj.width),
                y: clamp(p.y / imageObj.height)
            }));
        } else if (type === 'keypoints') {
            normalized.points = normalized.points.map(p => p ? {
                x: clamp(p.x / imageObj.width),
                y: clamp(p.y / imageObj.height),
                visible: p.visible
            } : null);
        }

        return normalized;
    });

    try {
        await apiRequest(`/images/${state.currentImage.id}/annotations`, {
            method: 'POST',
            body: JSON.stringify({
                data: normalizedAnnotations,
                created_by: 'human'
            })
        });

        // After saving, reload all annotations to get the updated combined view (background versioning)
        await loadAllAnnotationVersions(state.currentImage.id);

        // Mark as no longer dirty after successful save
        state.isDirty = false;

        // Instant Feedback: Update sidebar badge and state
        if (state.images) {
            const imgIndex = state.images.findIndex(img => img.id === state.currentImage.id);
            if (imgIndex !== -1) {
                state.images[imgIndex].verification_status = 'verified';
                state.images[imgIndex].has_annotations = true;
                state.images[imgIndex].latest_created_by = 'human';
            }
        }

        // Re-render the sidebar since verification status changed
        loadProjectImages(state.currentProject.id);

        console.log('Annotations saved successfully');
        if (notify === true || (typeof notify === 'object' && notify.type === 'click')) {
            showToast('Annotations saved', 'success');
        }

        // Check for auto-training and fine-tuning eligibility
        if (state.currentProject) {
            checkAutoTrainingEligibility();
        }

        return true;

    } catch (error) {
        console.error('Failed to save:', error);
        showToast('Failed to save annotations', 'error');
        return false;
    }
}

function saveAutoTrainSetting() {
    if (!state.currentProject) return;

    const trainEnabled = document.getElementById('auto-train-checkbox').checked;
    const finetuneEnabled = document.getElementById('auto-finetune-checkbox').checked;

    localStorage.setItem(`project_${state.currentProject.id}_auto_train`, String(trainEnabled));
    localStorage.setItem(`project_${state.currentProject.id}_auto_finetune`, String(finetuneEnabled));

    if (trainEnabled && finetuneEnabled) {
        showToast('Auto-training and Fine-tuning enabled', 'info');
    } else if (trainEnabled) {
        showToast('Auto-training enabled (Fine-tuning disabled)', 'info');
    } else if (finetuneEnabled) {
        showToast('Auto-training disabled (Fine-tuning enabled)', 'info');
    } else {
        showToast('All automation disabled for this project', 'info');
    }
}

// Auto-training detection and triggering
async function checkAutoTrainingEligibility() {
    try {
        // Only run for "fresh" training if enabled
        const autoTrainEnabled = localStorage.getItem(`project_${state.currentProject.id}_auto_train`) !== 'false';
        const autoFinetuneEnabled = localStorage.getItem(`project_${state.currentProject.id}_auto_finetune`) !== 'false';

        const result = await apiRequest(`/projects/${state.currentProject.id}/check-auto-train`);

        if (result.eligible && autoTrainEnabled && !state.activeTrainingTaskId) {
            const lastMilestone = localStorage.getItem(`project_${state.currentProject.id}_last_training_milestone`);

            if (lastMilestone !== String(result.verified_count)) {
                showToast(`🎓 Auto-training started! ${result.verified_count} verified images.`, 'info');
                localStorage.setItem(`project_${state.currentProject.id}_last_training_milestone`, String(result.verified_count));

                // Trigger training in background
                const epochs = result.has_custom_model ? 7 : 25;
                const trainResult = await apiRequest(`/projects/${state.currentProject.id}/train`, {
                    method: 'POST',
                    body: JSON.stringify({
                        epochs: epochs,
                        augment_multiplier: 10
                    })
                });

                if (trainResult.task_id) {
                    pollTrainingProgress(trainResult.task_id);
                }
            }
        }

        // Also check for fine-tuning - NOW respects autoFinetuneEnabled
        const ftResult = await apiRequest(`/projects/${state.currentProject.id}/correction-stats`);
        if (ftResult.eligible && autoFinetuneEnabled && !state.activeTrainingTaskId) {
            const lastFTMilestone = localStorage.getItem(`project_${state.currentProject.id}_last_finetune_milestone`);

            if (lastFTMilestone !== String(ftResult.correction_count)) {
                showToast(`🔧 Auto fine-tuning started! ${ftResult.correction_count} corrections.`, 'info');
                localStorage.setItem(`project_${state.currentProject.id}_last_finetune_milestone`, String(ftResult.correction_count));

                const ftTrainResult = await apiRequest(`/projects/${state.currentProject.id}/train`, {
                    method: 'POST',
                    body: JSON.stringify({
                        epochs: 7,
                        augment_multiplier: 10
                    })
                });

                if (ftTrainResult.task_id) {
                    pollTrainingProgress(ftTrainResult.task_id);
                }
            }
        }
    } catch (error) {
        console.error('Error checking training eligibility:', error);
    }
}

// Silent save for auto-save before navigation
async function saveAnnotationsIfNeeded() {
    return await saveAnnotations(false);
}

function pushToHistory() {
    // Mark annotations as modified
    state.isDirty = true;

    // Real undo/redo history stack
    if (!state.history) state.history = [];
    if (typeof state.historyIndex !== 'number') state.historyIndex = -1;

    // Truncate forward history when a new action is pushed
    if (state.historyIndex < state.history.length - 1) {
        state.history = state.history.slice(0, state.historyIndex + 1);
    }

    // Deep-clone the current annotations
    const snapshot = JSON.parse(JSON.stringify(state.tempAnnotations));
    state.history.push(snapshot);

    // Keep max 50 history entries
    if (state.history.length > 50) {
        state.history.shift();
    }
    state.historyIndex = state.history.length - 1;
}

function undoAnnotations() {
    if (!state.history || state.history.length === 0) return;
    if (state.historyIndex <= 0) {
        showToast('Nothing more to undo', 'info');
        return;
    }
    state.historyIndex--;
    state.tempAnnotations = JSON.parse(JSON.stringify(state.history[state.historyIndex]));
    state.selectedAnnotation = null;
    state.currentPolygon = null;
    state.isDirty = true;
    drawImage();
    renderInstancePanel();
    showToast('↩ Undo', 'info');
}

function redoAnnotations() {
    if (!state.history || state.historyIndex >= state.history.length - 1) {
        showToast('Nothing to redo', 'info');
        return;
    }
    state.historyIndex++;
    state.tempAnnotations = JSON.parse(JSON.stringify(state.history[state.historyIndex]));
    state.selectedAnnotation = null;
    state.currentPolygon = null;
    state.isDirty = true;
    drawImage();
    renderInstancePanel();
    showToast('↪ Redo', 'info');
}

// Drawing Functions
function setupCanvas() {
    // Mouse events for drawing
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('wheel', handleMouseWheel);
    canvas.addEventListener('dblclick', handleDoubleClick);
    canvas.addEventListener('contextmenu', handleContextMenu);

    // Resize listener
    window.addEventListener('resize', resizeCanvas);

    // ResizeObserver for more robust layout handling
    const resizeObserver = new ResizeObserver(() => {
        if (state.needsInitialFit) {
            resetZoomAndFit();
        } else {
            resizeCanvas();
        }
    });
    resizeObserver.observe(canvas.parentElement);

    resizeCanvas();
}

function resizeCanvas() {
    const container = canvas.parentElement;
    if (!container) return;

    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    drawImage();
}

function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    const displayX = (e.clientX - rect.left);
    const displayY = (e.clientY - rect.top);

    // Canvas width/height now match CSS pixels 1:1 since we resize it
    // So we just need to reverse pan and zoom to get to Image Coordinates
    return {
        x: (displayX - state.panOffset.x) / state.zoom,
        y: (displayY - state.panOffset.y) / state.zoom
    };
}

function handleMouseDown(e) {
    if (!imageObj) return;

    const pos = getMousePos(e);

    // Check for panning (middle click or space+left click or ctrl+left click)
    if (e.button === 1 || (e.button === 0 && (e.ctrlKey || state.currentTool === 'pan'))) {
        isPanning = true;
        lastPanPos = { x: e.clientX, y: e.clientY };
        canvas.style.cursor = 'grabbing';
        return;
    }

    // NEW Tool: SAM Single Point Click
    if (e.button === 0 && state.annotationType === 'polygon' && state.currentTool === 'auto-segment-click') {
        performClickSegmentation(pos, { x: e.clientX, y: e.clientY });
        return;
    }

    // LEFT CLICK - Unified selection and drawing
    if (e.button === 0) {
        // First: check if clicking on resize handle of selected annotation
        if (state.selectedAnnotation) {
            const handle = hitTestResizeHandles(pos, state.selectedAnnotation);
            if (handle) {
                state.isResizing = true;
                state.resizeHandle = handle;
                state.originalAnnotation = JSON.parse(JSON.stringify(state.selectedAnnotation));
                return;
            }

            // Check if clicking on a polygon point for point editing
            if ((state.selectedAnnotation.type === 'polygon' || state.annotationType === 'polygon') && state.selectedAnnotation.points) {
                const pointIndex = hitTestPolygonPoints(pos, state.selectedAnnotation);
                if (pointIndex !== -1) {
                    state.isDraggingPoint = true;
                    state.dragPointIndex = pointIndex;
                    state.originalAnnotation = JSON.parse(JSON.stringify(state.selectedAnnotation));
                    return;
                }

                // NEW: Optimized point addition - Single click on edge when already selected
                if (state.hoveredEdgePoint && state.hoveredEdgeIndex !== -1) {
                    const ann = state.selectedAnnotation;
                    // Add the point at the precise snapped location on the edge
                    ann.points.splice(state.hoveredEdgeIndex + 1, 0, state.hoveredEdgePoint);

                    // Immediately transition to dragging the new point
                    state.isDraggingPoint = true;
                    state.dragPointIndex = state.hoveredEdgeIndex + 1;
                    state.originalAnnotation = JSON.parse(JSON.stringify(ann));

                    // Clear hover state to prevent immediate re-addition
                    state.hoveredEdgePoint = null;
                    state.hoveredEdgeIndex = -1;

                    pushToHistory();
                    drawImage();
                    return;
                }
            }
        }

        // Second: check if clicking on an existing annotation
        // Pass current selection to allow cycling through overlaps
        const hitAnn = hitTestAnnotation(pos, state.selectedAnnotation);
        if (hitAnn) {
            if (hitAnn === state.selectedAnnotation) {
                // Already selected - start dragging the whole annotation
                state.isDragging = true;
                state.dragOffset = {
                    x: pos.x - (hitAnn.x || hitAnn.points[0].x),
                    y: pos.y - (hitAnn.y || hitAnn.points[0].y)
                };
                state.originalAnnotation = JSON.parse(JSON.stringify(hitAnn));
            } else {
                // Select this annotation and show class popup at click location
                selectAnnotation(hitAnn, { x: e.clientX, y: e.clientY });
            }
            return;
        }

        // Third: if we're in the middle of drawing a polygon, continue adding points
        if (state.annotationType === 'polygon' && state.currentPolygon) {
            const firstPoint = state.currentPolygon.points[0];
            const threshold = 15 / state.zoom;
            const dist = Math.sqrt(Math.pow(pos.x - firstPoint.x, 2) + Math.pow(pos.y - firstPoint.y, 2));

            if (dist < threshold && state.currentPolygon.points.length >= 3) {
                // Complete polygon
                state.currentPolygon.drawing = false;
                delete state.currentPolygon.tempPoint;
                const finishedPolygon = state.currentPolygon;
                state.currentPolygon = null;
                showClassSelectionPopup(finishedPolygon, { x: e.clientX, y: e.clientY });
            } else {
                state.currentPolygon.points.push(pos);
            }
            drawImage();
            return;
        }

        // Fourth: check if within image bounds for drawing
        if (pos.x < 0 || pos.x > imageObj.width || pos.y < 0 || pos.y > imageObj.height) {
            hideClassSelectionPopup();
            return;
        }

        // Deselect any selected annotation before drawing
        if (state.selectedAnnotation) {
            deselectAnnotation();
        }

        // Fifth: Start drawing new annotation
        if (state.annotationType === 'bbox' || (state.annotationType === 'polygon' && state.currentTool === 'auto-segment')) {
            state.isDrawing = true;
            state.tempAnnotations.push({
                type: 'bbox',
                class_id: state.currentClassId,
                x: pos.x,
                y: pos.y,
                width: 0,
                height: 0,
                drawing: true
            });
        } else if (state.annotationType === 'polygon' && state.currentTool === 'polygon') {
            // Start new polygon
            state.currentPolygon = {
                type: 'polygon',
                class_id: state.currentClassId,
                points: [pos],
                drawing: true
            };
            state.tempAnnotations.push(state.currentPolygon);
            drawImage();
        } else if (state.annotationType === 'keypoints') {
            const keypoint = { x: pos.x, y: pos.y, visible: true };
            let kpAnnotation = state.tempAnnotations.find(a => a.type === 'keypoints');
            if (!kpAnnotation) {
                const keypointCount = state.currentProject.label_schema.keypoints?.length || 0;
                kpAnnotation = {
                    type: 'keypoints',
                    class_id: state.currentClassId,
                    points: new Array(keypointCount).fill(null),
                    drawing: true
                };
                state.tempAnnotations.push(kpAnnotation);
            }
            if (state.currentKeypointIndex < kpAnnotation.points.length) {
                kpAnnotation.points[state.currentKeypointIndex] = keypoint;
            }
            drawImage();
        }
    }
    // Right-click is handled by handleContextMenu
}

// Hit test for polygon points - returns index of point clicked or -1
function hitTestPolygonPoints(pos, polygon) {
    if (!polygon || !polygon.points) return -1;

    const threshold = 16 / state.zoom; // Increased threshold to prioritize points over edges

    for (let i = 0; i < polygon.points.length; i++) {
        const point = polygon.points[i];
        const dist = Math.sqrt(Math.pow(pos.x - point.x, 2) + Math.pow(pos.y - point.y, 2));
        if (dist < threshold) {
            return i;
        }
    }
    return -1;
}

// Hit test for polygon edges - returns edge index (between point i and i+1) or -1
function hitTestPolygonEdges(pos, polygon) {
    if (!polygon || !polygon.points || polygon.points.length < 2) return -1;

    const threshold = 20 / state.zoom; // Increased threshold for easier selection

    for (let i = 0; i < polygon.points.length; i++) {
        const p1 = polygon.points[i];
        const p2 = polygon.points[(i + 1) % polygon.points.length];

        // Distance from point to line segment
        const dist = pointToSegmentDistance(pos, p1, p2);
        if (dist < threshold) {
            return i;
        }
    }
    return -1;
}

// Calculate distance from point to line segment
function pointToSegmentDistance(point, segStart, segEnd) {
    const dx = segEnd.x - segStart.x;
    const dy = segEnd.y - segStart.y;
    const lenSq = dx * dx + dy * dy;

    if (lenSq === 0) return Math.sqrt(Math.pow(point.x - segStart.x, 2) + Math.pow(point.y - segStart.y, 2));

    let t = ((point.x - segStart.x) * dx + (point.y - segStart.y) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));

    const closestX = segStart.x + t * dx;
    const closestY = segStart.y + t * dy;

    return Math.sqrt(Math.pow(point.x - closestX, 2) + Math.pow(point.y - closestY, 2));
}

function getClosestPointOnSegment(point, segStart, segEnd) {
    const dx = segEnd.x - segStart.x;
    const dy = segEnd.y - segStart.y;
    const lenSq = dx * dx + dy * dy;

    if (lenSq === 0) return { x: segStart.x, y: segStart.y };

    let t = ((point.x - segStart.x) * dx + (point.y - segStart.y) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));

    return {
        x: segStart.x + t * dx,
        y: segStart.y + t * dy
    };
}

function handleMouseMove(e) {
    if (!imageObj) return;

    if (isPanning) {
        const dx = e.clientX - lastPanPos.x;
        const dy = e.clientY - lastPanPos.y;

        state.panOffset.x += dx;
        state.panOffset.y += dy;

        lastPanPos = { x: e.clientX, y: e.clientY };
        drawImage();
        return;
    }

    const pos = getMousePos(e);

    // Handle polygon point dragging
    if (state.isDraggingPoint && state.selectedAnnotation && state.dragPointIndex !== -1) {
        state.selectedAnnotation.points[state.dragPointIndex] = { x: pos.x, y: pos.y };
        drawImage();
        return;
    }

    // Handle annotation dragging
    if (state.isDragging && state.selectedAnnotation) {
        const ann = state.selectedAnnotation;
        if (ann.type === 'bbox' || state.annotationType === 'bbox') {
            ann.x = pos.x - state.dragOffset.x;
            ann.y = pos.y - state.dragOffset.y;
        } else if (ann.type === 'polygon' || state.annotationType === 'polygon') {
            const origFirstPoint = state.originalAnnotation.points[0];
            const dx = pos.x - state.dragOffset.x - origFirstPoint.x;
            const dy = pos.y - state.dragOffset.y - origFirstPoint.y;
            ann.points = state.originalAnnotation.points.map(p => ({
                x: p.x + dx,
                y: p.y + dy
            }));
        }
        drawImage();
        return;
    }

    // Handle resizing
    if (state.isResizing && state.selectedAnnotation && state.resizeHandle) {
        const ann = state.selectedAnnotation;
        const orig = state.originalAnnotation;
        const handle = state.resizeHandle;

        if (ann.type === 'bbox' || state.annotationType === 'bbox') {
            switch (handle) {
                case 'nw':
                    ann.width = orig.x + orig.width - pos.x;
                    ann.height = orig.y + orig.height - pos.y;
                    ann.x = pos.x;
                    ann.y = pos.y;
                    break;
                case 'ne':
                    ann.width = pos.x - orig.x;
                    ann.height = orig.y + orig.height - pos.y;
                    ann.y = pos.y;
                    break;
                case 'sw':
                    ann.width = orig.x + orig.width - pos.x;
                    ann.height = pos.y - orig.y;
                    ann.x = pos.x;
                    break;
                case 'se':
                    ann.width = pos.x - orig.x;
                    ann.height = pos.y - orig.y;
                    break;
                case 'n':
                    ann.height = orig.y + orig.height - pos.y;
                    ann.y = pos.y;
                    break;
                case 's':
                    ann.height = pos.y - orig.y;
                    break;
                case 'w':
                    ann.width = orig.x + orig.width - pos.x;
                    ann.x = pos.x;
                    break;
                case 'e':
                    ann.width = pos.x - orig.x;
                    break;
            }
        }
        drawImage();
        return;
    }

    // Update cursor based on what's under it
    if (state.selectedAnnotation) {
        // Check for polygon point
        if ((state.selectedAnnotation.type === 'polygon' || state.annotationType === 'polygon') && state.selectedAnnotation.points) {
            const pointIndex = hitTestPolygonPoints(pos, state.selectedAnnotation);
            if (pointIndex !== -1) {
                canvas.style.cursor = 'grab';
                // Clear ghost point if we are hovering a real point
                if (state.hoveredEdgePoint) {
                    state.hoveredEdgePoint = null;
                    state.hoveredEdgeIndex = -1;
                    drawImage();
                }
                return;
            }
        }
        // Check for resize handle
        const handle = hitTestResizeHandles(pos, state.selectedAnnotation);
        if (handle) {
            canvas.style.cursor = getResizeCursor(handle);
            state.hoveredEdgePoint = null;
            state.hoveredEdgeIndex = -1;
            return;
        }

        // Check for edge hover (for adding points)
        if ((state.selectedAnnotation.type === 'polygon' || state.annotationType === 'polygon') && state.selectedAnnotation.points) {
            const edgeIndex = hitTestPolygonEdges(pos, state.selectedAnnotation);
            if (edgeIndex !== -1) {
                const p1 = state.selectedAnnotation.points[edgeIndex];
                const p2 = state.selectedAnnotation.points[(edgeIndex + 1) % state.selectedAnnotation.points.length];
                state.hoveredEdgePoint = getClosestPointOnSegment(pos, p1, p2);
                state.hoveredEdgeIndex = edgeIndex;
                canvas.style.cursor = 'copy'; // Indicates "Add"
                drawImage(); // Redraw to show ghost point
                return;
            }
        }
    }

    // Reset edge hover if not found
    if (state.hoveredEdgePoint) {
        state.hoveredEdgePoint = null;
        state.hoveredEdgeIndex = -1;
        drawImage();
    }

    // Check if hovering over any annotation
    const hitAnn = hitTestAnnotation(pos);
    if (hitAnn) {
        canvas.style.cursor = hitAnn === state.selectedAnnotation ? 'move' : 'pointer';
    } else {
        canvas.style.cursor = 'crosshair';
    }

    // DRAW MODE - polygon temp line
    const isBBoxMode = state.annotationType === 'bbox' || (state.annotationType === 'polygon' && state.currentTool === 'auto-segment');
    if (state.isDrawing && isBBoxMode) {
        const currentBox = state.tempAnnotations.find(a => a.drawing);
        if (currentBox) {
            currentBox.width = pos.x - currentBox.x;
            currentBox.height = pos.y - currentBox.y;
            drawImage();
        }
    } else if (state.annotationType === 'polygon' && state.currentPolygon) {
        state.currentPolygon.tempPoint = pos;
        drawImage();
    }
}

// Helper to get cursor for resize handles
function getResizeCursor(handle) {
    const cursors = {
        'nw': 'nwse-resize',
        'se': 'nwse-resize',
        'ne': 'nesw-resize',
        'sw': 'nesw-resize',
        'n': 'ns-resize',
        's': 'ns-resize',
        'e': 'ew-resize',
        'w': 'ew-resize'
    };
    return cursors[handle] || 'default';
}

async function convertBoxToPolygon(box, screenPos) {
    try {
        showToast('🪄 Segmenting object...', 'info');

        const response = await apiRequest(`/images/${state.currentImage.id}/segment`, {
            method: 'POST',
            body: JSON.stringify({
                x: box.x,
                y: box.y,
                width: box.width,
                height: box.height
            })
        });

        // Remove the temporary bbox
        state.tempAnnotations = state.tempAnnotations.filter(a => a !== box);

        // Add the new polygon
        const newPolygon = {
            type: 'polygon',
            class_id: box.class_id,
            points: response.points.map(p => ({
                x: p.x * imageObj.width,
                y: p.y * imageObj.height
            })),
            source: 'sam'
        };

        state.tempAnnotations.push(newPolygon);
        drawImage();

        // Show class selector for the new polygon
        showClassSelectionPopup(newPolygon, screenPos);

    } catch (error) {
        console.error('Segmentation failed:', error);
        showToast('Segmentation failed: ' + error.message, 'error');
        // Keep the bbox if SAM fails as a fallback
        box.drawing = false;
        showClassSelectionPopup(box, screenPos);
    }
}

async function performClickSegmentation(pos, screenPos) {
    try {
        showToast('🪄 Segmenting from click...', 'info');

        const response = await apiRequest(`/images/${state.currentImage.id}/segment`, {
            method: 'POST',
            body: JSON.stringify({
                x: pos.x,
                y: pos.y
            })
        });

        // Add the new polygon
        const newPolygon = {
            type: 'polygon',
            class_id: state.currentClassId,
            points: response.points.map(p => ({
                x: p.x * imageObj.width,
                y: p.y * imageObj.height
            })),
            source: 'sam'
        };

        state.tempAnnotations.push(newPolygon);
        selectAnnotation(newPolygon, screenPos);
        drawImage();

    } catch (error) {
        console.error('Click segmentation failed:', error);
        showToast('Click segmentation failed: ' + error.message, 'error');
    }
}

function handleMouseUp(e) {
    if (isPanning) {
        isPanning = false;
        canvas.style.cursor = 'default';
        return;
    }

    // Finalize polygon point dragging
    if (state.isDraggingPoint) {
        pushToHistory();
        state.isDraggingPoint = false;
        state.dragPointIndex = -1;
        state.originalAnnotation = null;
        drawImage();
        return;
    }

    // Finalize drag/resize
    if (state.isDragging || state.isResizing) {
        // Normalize bbox if width/height became negative
        if (state.selectedAnnotation && (state.selectedAnnotation.type === 'bbox' || state.annotationType === 'bbox')) {
            const ann = state.selectedAnnotation;
            if (ann.width < 0) {
                ann.x += ann.width;
                ann.width = -ann.width;
            }
            if (ann.height < 0) {
                ann.y += ann.height;
                ann.height = -ann.height;
            }
        }
        pushToHistory();
        state.isDragging = false;
        state.isResizing = false;
        state.resizeHandle = null;
        state.originalAnnotation = null;
        drawImage();
        return;
    }

    // DRAW MODE - finalize bbox
    if (!state.isDrawing) return;

    if (state.annotationType === 'bbox' || (state.annotationType === 'polygon' && state.currentTool === 'auto-segment')) {
        const currentBox = state.tempAnnotations.find(a => a.drawing);
        if (currentBox) {
            // Remove if too small
            if (Math.abs(currentBox.width) < 5 || Math.abs(currentBox.height) < 5) {
                state.tempAnnotations = state.tempAnnotations.filter(a => a !== currentBox);
            } else {
                // Ensure positive width/height
                if (currentBox.width < 0) {
                    currentBox.x += currentBox.width;
                    currentBox.width = -currentBox.width;
                }
                if (currentBox.height < 0) {
                    currentBox.y += currentBox.height;
                    currentBox.height = -currentBox.height;
                }
                currentBox.drawing = false;

                // SPECIAL: If in polygon project and using auto-segment, trigger SAM
                if (state.annotationType === 'polygon' && state.currentTool === 'auto-segment') {
                    convertBoxToPolygon(currentBox, { x: e.clientX, y: e.clientY });
                } else {
                    showClassSelectionPopup(currentBox, { x: e.clientX, y: e.clientY });
                }
            }
        }
    }

    state.isDrawing = false;
    drawImage();
}

// Right-click handler for deleting polygon points
function handleContextMenu(e) {
    e.preventDefault();

    if (!imageObj) return;

    const pos = getMousePos(e);

    // Check if right-clicking on a polygon point while that polygon is selected
    if (state.selectedAnnotation &&
        (state.selectedAnnotation.type === 'polygon' || state.annotationType === 'polygon') &&
        state.selectedAnnotation.points) {

        const pointIndex = hitTestPolygonPoints(pos, state.selectedAnnotation);

        if (pointIndex !== -1) {
            // Need at least 3 points for a valid polygon
            if (state.selectedAnnotation.points.length > 3) {
                state.selectedAnnotation.points.splice(pointIndex, 1);
                pushToHistory();
                drawImage();
            } else {
                // Can't delete - show feedback
                console.log('Cannot delete point: polygon must have at least 3 points');
            }
            return;
        }
    }

    // Default behavior: finish polygon drawing if in progress
    if (state.annotationType === 'polygon' && state.currentPolygon && state.currentPolygon.points.length >= 3) {
        state.currentPolygon.drawing = false;
        delete state.currentPolygon.tempPoint;
        const finishedPolygon = state.currentPolygon;
        state.currentPolygon = null;
        showClassSelectionPopup(finishedPolygon, { x: e.clientX, y: e.clientY });
        drawImage();
    }
}

// Double-click to add a new point to polygon edge
function handleDoubleClick(e) {
    if (!imageObj || !state.selectedAnnotation) return;

    const pos = getMousePos(e);
    const ann = state.selectedAnnotation;

    // Only for polygons - Points are now added via single click in handleMouseDown for better UX
    // We keep this as a no-op to prevent duplicate points from a double-click
    if ((ann.type === 'polygon' || state.annotationType === 'polygon') && ann.points) {
        // Point addition is now handled in handleMouseDown
        return;
    }
}

function handleMouseWheel(e) {
    e.preventDefault();

    const zoomIntensity = 0.1;
    const isZoomIn = e.deltaY < 0;
    const zoomFactor = isZoomIn ? (1 + zoomIntensity) : (1 - zoomIntensity);

    const oldZoom = state.zoom;
    const newZoom = state.zoom * zoomFactor;

    // Clamp zoom
    if (newZoom < 0.1 || newZoom > 20) return;

    // Zoom towards cursor
    const rect = canvas.getBoundingClientRect();

    // Get mouse position relative to canvas (which is now 1:1 with container pixels)
    const canvasX = (e.clientX - rect.left);
    const canvasY = (e.clientY - rect.top);

    // Map canvas coordinates to "world" (image) coordinates using CURRENT zoom/pan
    const worldMouseX = (canvasX - state.panOffset.x) / oldZoom;
    const worldMouseY = (canvasY - state.panOffset.y) / oldZoom;

    // Update state
    state.zoom = newZoom;

    // New zoom/pan should keep the "world" point under the same "canvas" point
    state.panOffset.x = canvasX - worldMouseX * newZoom;
    state.panOffset.y = canvasY - worldMouseY * newZoom;

    updateZoomDisplay();
    drawImage();
}

function drawImage() {
    // Safety check: Don't attempt to draw if imageObj is missing or "broken"
    if (!imageObj || !imageObj.complete || imageObj.naturalWidth === 0) {
        return;
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.save();

    // Apply pan and zoom
    ctx.translate(state.panOffset.x, state.panOffset.y);
    ctx.scale(state.zoom, state.zoom);

    // Draw image
    ctx.drawImage(imageObj, 0, 0);

    // Draw existing annotations
    if (state.currentAnnotations && state.currentAnnotations.length > 0) {
        drawAnnotationData(state.currentAnnotations, false);
    }

    // Draw temporary annotations
    drawAnnotationData(state.tempAnnotations, true);

    // Draw selection handles for selected annotation
    if (state.selectedAnnotation) {
        drawSelectionHandles(state.selectedAnnotation);
    }

    ctx.restore();

    // Draw Null Watermark if image is marked as processed but has no annotations
    // This provides the visual feedback requested by the user.
    if (state.currentImage && state.currentImage.status === 'processed' &&
        state.tempAnnotations.length === 0 &&
        (!state.currentAnnotations || state.currentAnnotations.length === 0)) {

        ctx.save();

        // Large, semi-transparent red watermark
        ctx.font = 'bold 70px Arial';
        ctx.fillStyle = 'rgba(255, 69, 58, 0.4)'; // Clean semi-transparent red
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Center text on canvas
        ctx.fillText('NULL IMAGE', canvas.width / 2, canvas.height / 2 - 20);

        ctx.font = 'bold 24px Arial';
        ctx.fillText('(NEGATIVE SAMPLE)', canvas.width / 2, canvas.height / 2 + 40);

        // Subtle border to reinforce the "checked/empty" state
        ctx.strokeStyle = 'rgba(255, 69, 58, 0.3)';
        ctx.lineWidth = 10;
        ctx.strokeRect(40, 40, canvas.width - 80, canvas.height - 80);

        ctx.restore();
    }

    // After all rendering, refresh instance panel (debounced)
    if (!drawImage._panelTimer) {
        drawImage._panelTimer = setTimeout(() => {
            renderInstancePanel();
            drawImage._panelTimer = null;
        }, 50);
    }
}

function drawAnnotationData(annotations, isTemp) {
    annotations.forEach(ann => {
        if (ann._hidden) return; // Skip hidden annotations
        if (ann.type === 'bbox' || state.annotationType === 'bbox') {
            drawBoundingBox(ann, isTemp);
        } else if (ann.type === 'polygon' || state.annotationType === 'polygon') {
            drawPolygon(ann, isTemp);
        } else if (ann.type === 'keypoints' || state.annotationType === 'keypoints') {
            drawKeypoints(ann, isTemp);
        }
    });
}


function drawBoundingBox(box, isTemp) {
    const color = getClassColor(box.class_id);
    ctx.strokeStyle = isTemp && box.drawing ? '#FF3333' : color;
    ctx.lineWidth = 4;
    ctx.setLineDash(isTemp && box.drawing ? [8, 4] : []);

    ctx.strokeRect(box.x, box.y, box.width, box.height);

    // Draw class label
    if (!box.drawing) {
        let label = getClassName(box.class_id);
        if (box.source_version) {
            label += ` (V${box.source_version})`;
        }

        ctx.fillStyle = color;
        ctx.globalAlpha = 0.7;
        ctx.fillRect(box.x, box.y - 20, ctx.measureText(label).width + 10, 20);
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.fillText(label, box.x + 5, box.y - 5);
    }

    ctx.setLineDash([]);
}

function drawPolygon(polygon, isTemp) {
    if (!polygon.points || polygon.points.length === 0) return;

    const color = getClassColor(polygon.class_id);
    ctx.strokeStyle = isTemp && polygon.drawing ? '#FF3333' : color;
    ctx.lineWidth = 4;
    ctx.setLineDash(isTemp && polygon.drawing ? [8, 4] : []);

    ctx.beginPath();
    ctx.moveTo(polygon.points[0].x, polygon.points[0].y);

    for (let i = 1; i < polygon.points.length; i++) {
        ctx.lineTo(polygon.points[i].x, polygon.points[i].y);
    }

    // Draw temporary line to mouse
    if (polygon.drawing && polygon.tempPoint) {
        ctx.lineTo(polygon.tempPoint.x, polygon.tempPoint.y);
    } else if (polygon.points.length > 2 && !polygon.drawing) {
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.2;
        ctx.fill();
        ctx.globalAlpha = 1.0;
    }

    ctx.stroke();

    // Draw points
    polygon.points.forEach((point, idx) => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = isTemp && polygon.drawing ? '#FF0000' : color;
        ctx.fill();
    });

    // Draw version label for polygon if available
    if (polygon.source_version && polygon.points.length > 0) {
        const firstPoint = polygon.points[0];
        const label = `V${polygon.source_version}`;
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.7;
        ctx.fillRect(firstPoint.x, firstPoint.y - 20, ctx.measureText(label).width + 10, 20);
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = 'white';
        ctx.font = '10px Arial';
        ctx.fillText(label, firstPoint.x + 5, firstPoint.y - 8);
    }

    ctx.setLineDash([]);
}

function drawKeypoints(keypointsAnn, isTemp) {
    if (!keypointsAnn.points) return;

    const color = getClassColor(keypointsAnn.class_id);

    keypointsAnn.points.forEach((point, index) => {
        if (!point) return;

        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = point.visible ? color : '#888888';
        ctx.fill();

        // Draw keypoint label
        ctx.fillStyle = 'white';
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 3;
        ctx.font = 'bold 12px Arial';
        ctx.strokeText(`${index}`, point.x + 8, point.y - 8);
        ctx.fillText(`${index}`, point.x + 8, point.y - 8);
    });
}

function drawAnnotations() {
    drawImage();
}

function getClassColor(classId) {
    if (state.classes && state.classes[classId]) {
        return state.classes[classId].color || '#00D9FF';
    }
    // Brighter, more vibrant colors for better visibility
    const colors = ['#00D9FF', '#FF3366', '#00FF88', '#FFD700', '#FF66FF', '#00FFFF'];
    return colors[classId % colors.length];
}

function getClassName(classId) {
    if (state.classes && state.classes[classId]) {
        return state.classes[classId].name || `Class ${classId}`;
    }
    return `Class ${classId}`;
}

// UI Functions
function updateToolbar() {
    const toolbar = document.getElementById('toolbar');
    toolbar.innerHTML = '';

    if (!state.annotationType) return;

    if (state.annotationType === 'bbox') {
        toolbar.innerHTML = `
            <div style="display: flex; gap: 10px; align-items: center; width: 100%; justify-content: space-between;">
                <div style="display: flex; gap: 10px; align-items: center;">
                    <button class="btn btn-primary" onclick="setTool('bbox')">
                        Draw Bounding Box
                    </button>
                    <button class="btn btn-danger" onclick="clearAnnotations()">
                        Clear All
                    </button>
                    <div style="display: flex; gap: 5px; align-items: center;">
                        <select class="select-control" id="class-selector-bbox" onchange="setCurrentClass(this.value)" style="width: 150px;">
                            ${state.classes.map(cls =>
            `<option value="${cls.id}" ${state.currentClassId === cls.id ? 'selected' : ''}>${cls.name}</option>`
        ).join('')}
                        </select>
                        <button class="btn btn-secondary" onclick="addClassWhileAnnotating()" title="Add New Class">+</button>
                        <button class="btn btn-secondary" onclick="showEditClassModal(state.currentClassId)" title="Edit Selected Class">✏️</button>
                    </div>
                </div>
                <div style="display: flex; gap: 10px;">
                     <button class="btn btn-warning" onclick="markAsNull()" title="Mark as Null (No Objects)" style="background: #f1c40f; border: none;">
                        🚫 Null Image
                    </button>
                    <button class="btn btn-danger" onclick="deleteImage()" title="Permanently Delete Image">
                        🗑️ Delete
                    </button>
                </div>
            </div>
        `;
    } else if (state.annotationType === 'polygon') {
        const isAutoSeg = state.currentTool === 'auto-segment' || !state.currentTool || state.currentTool === 'polygon';
        if (!state.currentTool) state.currentTool = 'auto-segment'; // Default to auto-segment

        toolbar.innerHTML = `
            <div style="display: flex; gap: 10px; align-items: center; width: 100%; justify-content: space-between;">
                <div style="display: flex; gap: 10px; align-items: center;">
                    <div class="tool-group" style="display: flex; background: #f8f9fa; padding: 3px; border-radius: 6px; border: 1px solid #dee2e6;">
                        <button class="btn ${state.currentTool === 'auto-segment' ? 'btn-primary' : 'btn-light'}" 
                            onclick="setTool('auto-segment')" title="Draw a box to segment an object" style="padding: 6px 12px; font-size: 13px; border-radius: 4px;">
                            🪄 Draw Box
                        </button>
                        <button class="btn ${state.currentTool === 'auto-segment-click' ? 'btn-primary' : 'btn-light'}" 
                            onclick="setTool('auto-segment-click')" title="Single click an object to segment it" style="padding: 6px 12px; font-size: 13px; border-radius: 4px;">
                            🪄 Click
                        </button>
                        <button class="btn ${state.currentTool === 'polygon' ? 'btn-primary' : 'btn-light'}" 
                            onclick="setTool('polygon')" title="Manually draw polygon points" style="padding: 6px 12px; font-size: 13px; border-radius: 4px;">
                            ⬡ Manual
                        </button>
                    </div>
                    <button class="btn btn-danger" onclick="clearAnnotations()">
                        Clear All
                    </button>
                    <div style="display: flex; gap: 5px; align-items: center;">
                        <select class="select-control" id="class-selector-polygon" onchange="setCurrentClass(this.value)" style="width: 150px;">
                             ${state.classes.map(cls =>
            `<option value="${cls.id}" ${state.currentClassId === cls.id ? 'selected' : ''}>${cls.name}</option>`
        ).join('')}
                        </select>
                        <button class="btn btn-secondary" onclick="addClassWhileAnnotating()" title="Add New Class">+</button>
                        <button class="btn btn-secondary" onclick="showEditClassModal(state.currentClassId)" title="Edit Selected Class">✏️</button>
                    </div>
                </div>
                 <div style="display: flex; gap: 10px;">
                     <button class="btn btn-warning" onclick="markAsNull()" title="Mark as Null (No Objects)" style="background: #f1c40f; border: none;">
                        🚫 Null Image
                    </button>
                    <button class="btn btn-danger" onclick="deleteImage()" title="Permanently Delete Image">
                        🗑️ Delete
                    </button>
                </div>
            </div>
        `;
    } else if (state.annotationType === 'keypoints') {
        const keypoints = state.currentProject?.label_schema?.keypoints || [];
        toolbar.innerHTML = `
            <button class="btn btn-primary" onclick="setTool('keypoints')">
                Annotate Keypoints
            </button>
            <button class="btn btn-danger" onclick="clearAnnotations()">
                Clear All
            </button>
            <select class="select-control" onchange="setCurrentKeypointIndex(this.value)" style="width: 200px;">
                ${keypoints.map((kp, i) =>
            `<option value="${i}">${kp.name || `Keypoint ${i}`}</option>`
        ).join('')}
            </select>
            <div style="display: flex; gap: 5px; align-items: center; margin-left: 10px;">
                <select class="select-control" id="class-selector-keypoints" onchange="setCurrentClass(this.value)" style="width: 150px;">
                    ${state.classes.map(cls =>
            `<option value="${cls.id}" ${state.currentClassId === cls.id ? 'selected' : ''}>${cls.name}</option>`
        ).join('')}
                </select>
                <button class="btn btn-secondary" onclick="addClassWhileAnnotating()" title="Add New Class">+</button>
                <button class="btn btn-secondary" onclick="showEditClassModal(state.currentClassId)" title="Edit Selected Class">✏️</button>
            </div>
        `;
    }


    // Add save button if image is loaded
    if (state.currentImage) {
        const isVerified = state.currentImage.latest_annotation && state.currentImage.latest_annotation.created_by === 'human';
        const isAI = state.currentImage.latest_annotation && state.currentImage.latest_annotation.created_by === 'auto';

        const saveBtn = document.createElement('button');
        saveBtn.className = isVerified ? 'btn btn-success' : 'btn btn-primary';
        saveBtn.innerHTML = isVerified ? '✅ Save & Verify' : '💾 Save Annotations';
        saveBtn.onclick = () => saveAnnotations();
        saveBtn.style.marginLeft = 'auto';
        saveBtn.style.minWidth = '180px';
        saveBtn.title = isVerified ? "Already verified by you" : "Save and mark as verified";
        toolbar.appendChild(saveBtn);

        if (isAI || (state.tempAnnotations.length > 0 && !isVerified)) {
            const confirmBtn = document.createElement('button');
            confirmBtn.className = 'btn btn-success';
            confirmBtn.innerHTML = '🚀 Confirm & Next';
            confirmBtn.style.marginLeft = '5px';
            confirmBtn.onclick = confirmAndNext;
            confirmBtn.style.minWidth = '160px';
            confirmBtn.title = "Save these annotations as verified and move to next image";
            toolbar.appendChild(confirmBtn);
        }
    }
}

function changeSelectedAnnotationClass(newClassId) {
    if (state.selectedAnnotation) {
        state.selectedAnnotation.class_id = parseInt(newClassId);
        pushToHistory();
        drawImage();
        // Refresh toolbar to reflect any color changes if needed
        updateToolbar();
    }
}

function updateGlobalActions() {
    const actions = document.getElementById('global-actions');
    if (!actions) return;

    if (state.currentProject) {

        let importButtonHTML = '';
        if (state.user && (state.user.role === 'admin' || state.user.role === 'sub_admin')) {
            importButtonHTML = `
            <button class="btn btn-primary" onclick="showImportModal()" title="Import Images or Videos" style="font-size: 12px; padding: 5px 12px;">
                📥 Import
            </button>
            <div style="width: 1px; height: 22px; background: rgba(255,255,255,0.2); margin: 0 4px;"></div>`;
        }

        actions.innerHTML = `
            ${importButtonHTML}
            <div style="display: flex; align-items: center; gap: 5px; background: rgba(255,255,255,0.06); padding: 3px 6px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);" title="AI Annotation Tools">
                <span style="font-size: 10px; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.5px; padding-right: 4px;">AI</span>
                <button class="btn" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-size: 12px; padding: 4px 10px;" onclick="runAutoAnnotate()" title="AI Auto-Annotate current image (uses YOLO-World + Grounding DINO ensemble)">
                    🤖 Annotate
                </button>
                <button class="btn" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; font-size: 12px; padding: 4px 10px;" onclick="runBatchAutoAnnotate()" title="Batch AI: Auto-annotate all pending images in the background">
                    🚀 Batch
                </button>
                <label style="font-size: 11px; font-weight: 600; margin: 0 4px; cursor: pointer; display: flex; align-items: center; gap: 4px; color: rgba(255,255,255,0.8);" title="Auto-AI: Automatically run AI annotation every time you switch to a new image">
                    <input type="checkbox" id="auto-ai-toggle" ${state.autoAI ? 'checked' : ''} onchange="toggleAutoAI(this.checked)" style="margin: 0;">
                    <span>Auto</span>
                </label>
            </div>
            <div style="width: 1px; height: 22px; background: rgba(255,255,255,0.2); margin: 0 4px;"></div>
            <button class="btn" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: #1a1a1a; font-size: 12px; padding: 5px 12px; font-weight: 700;" onclick="showTrainingModal()" title="Train a custom YOLO model on your verified human annotations">
                🎓 Train
            </button>
        `;
    } else {
        actions.innerHTML = '';
    }
}

function toggleAutoAI(enabled) {
    state.autoAI = enabled;
    showToast(`Streaming AI ${enabled ? 'Enabled' : 'Disabled'}`, enabled ? 'success' : 'info');
}

// --- AI Auto-Annotation Functions ---
async function runAutoAnnotate() {
    if (!state.currentImage) {
        showToast('Please select an image first', 'warning');
        return;
    }

    const progressBar = document.getElementById('ai-progress-toolbar');
    const progressFill = document.getElementById('ai-progress-fill-toolbar');
    const progressText = document.getElementById('ai-progress-text-toolbar');

    if (progressBar) {
        progressBar.style.display = 'block';
        if (progressFill) progressFill.style.width = '30%';
        if (progressText) {
            progressText.style.display = 'inline-block';
            progressText.textContent = '🤖 AI: Thinking...';
        }
    }

    showToast('🤖 Running AI auto-annotation...', 'info');

    try {
        if (progressFill) progressFill.style.width = '60%';

        const result = await apiRequest(`/images/${state.currentImage.id}/auto-annotate`, {
            method: 'POST',
            body: JSON.stringify({
                use_yolo_world: true,
                use_grounding_dino: true,
                min_confidence: 0.3,
                nms_iou_threshold: 0.5
            })
        });

        if (progressBar) progressBar.style.display = 'none';
        if (progressText) progressText.style.display = 'none';

        if (result.count > 0) {
            // Determine which models were used for feedback
            let modelFeedback = "Zero-shot Base Models";
            if (result.custom_model_active) {
                modelFeedback = "Custom Trained YOLO Model";
            } else if (result.models_used.length > 0) {
                const models = result.models_used.map(m => m === 'yolo_world' ? 'YOLO-World' : (m === 'grounding_dino' ? 'Grounding DINO' : m));
                modelFeedback = `Base Models (${models.join(', ')})`;
            }

            // Convert normalized annotations to image coordinates
            const newAnnotations = result.annotations.map(ann => {
                const denormalized = { ...ann };
                if (ann.points) {
                    denormalized.points = ann.points.map(p => ({
                        x: p.x * imageObj.width,
                        y: p.y * imageObj.height
                    }));
                } else {
                    denormalized.x = ann.x * imageObj.width;
                    denormalized.y = ann.y * imageObj.height;
                    denormalized.width = ann.width * imageObj.width;
                    denormalized.height = ann.height * imageObj.height;
                }
                return denormalized;
            });

            // Replace existing annotations instead of appending to avoid overlapping
            state.tempAnnotations = newAnnotations;
            state.isDirty = true;
            drawImage();

            showToast(`✅ Found ${result.count} objects using ${modelFeedback}`, 'success');
        } else {
            showToast('No objects detected', 'warning');
        }

    } catch (error) {
        console.error('Auto-annotation failed:', error);
        if (progressBar) progressBar.style.display = 'none';
        if (progressText) progressText.style.display = 'none';
        showToast('Auto-annotation failed. Check console for details.', 'error');
    }
}

async function runBatchAutoAnnotate() {
    if (!state.currentProject) {
        showToast('No project selected', 'warning');
        return;
    }

    if (!confirm('This will auto-annotate all pending images in the background. Continue?')) {
        return;
    }

    showToast('🚀 Starting batch auto-annotation...', 'info');

    try {
        const result = await apiRequest(`/projects/${state.currentProject.id}/auto-annotate-batch`, {
            method: 'POST',
            body: JSON.stringify({
                use_yolo_world: true,
                use_grounding_dino: true,
                min_confidence: 0.3,
                nms_iou_threshold: 0.5
            })
        });

        showToast(`Started! Processing ${result.total} images in background.`, 'success');

        // Start polling for progress
        if (result.task_id) {
            state.activeBatchTaskId = result.task_id;  // Store in global state
            pollBatchProgress(result.task_id);
        }

    } catch (error) {
        console.error('Batch auto-annotation failed:', error);
        showToast('Batch auto-annotation failed', 'error');
    }
}

async function pollBatchProgress(taskId) {
    const progressBar = document.getElementById('ai-progress-toolbar');
    const progressFill = document.getElementById('ai-progress-fill-toolbar');
    const progressText = document.getElementById('ai-progress-text-toolbar');

    const checkProgress = async () => {
        try {
            const response = await fetch(`${API_BASE}/tasks/${taskId}`);
            if (!response.ok) {
                if (response.status === 404) {
                    console.warn('Batch task not found, stopping poll.');
                    if (progressBar) progressBar.style.display = 'none';
                    return;
                }
                throw new Error(`Task Error: ${response.status}`);
            }
            const task = await response.json();

            if (task.status === 'processing') {
                if (progressBar) progressBar.style.display = 'block';
                if (progressFill) progressFill.style.width = `${task.progress}%`;
                if (progressText) {
                    progressText.style.display = 'inline-block';
                    const msg = task.message || `Batch: ${task.progress}%`;
                    progressText.textContent = `🤖 ${msg} (${task.progress}%)`;
                }

                // Global Monitor Update
                const globalMonitor = document.getElementById('global-task-monitor');
                const globalFill = document.getElementById('global-task-fill');
                const globalPercent = document.getElementById('global-task-percent');
                const globalLabel = document.getElementById('global-task-label');
                const globalIcon = document.getElementById('global-task-icon');

                if (globalMonitor) {
                    globalMonitor.style.display = 'flex';
                    if (globalFill) globalFill.style.width = `${task.progress}%`;
                    if (globalPercent) globalPercent.textContent = `${task.progress}%`;
                    if (globalLabel) globalLabel.textContent = (task.message || 'BATCH PROCESSING...').toUpperCase();
                    if (globalIcon) globalIcon.textContent = '🤖';
                }

                // Refresh image list to show newly annotated images
                if (task.completed_images && task.completed_images.length > 0) {
                    loadProjectImages(state.currentProject.id);
                }

                setTimeout(checkProgress, 3000);
            } else if (task.status === 'completed') {
                showToast(`✅ Batch complete! Annotated ${task.total} images.`, 'success');
                if (progressBar) progressBar.style.display = 'none';
                if (progressText) progressText.style.display = 'none';

                const globalMonitor = document.getElementById('global-task-monitor');
                if (globalMonitor) globalMonitor.style.display = 'none';

                state.activeBatchTaskId = null;  // Clear active task
                loadProjectImages(state.currentProject.id);
            } else if (task.status === 'failed') {
                showToast(`❌ Batch failed: ${task.error}`, 'error');
                if (progressBar) progressBar.style.display = 'none';
                if (progressText) progressText.style.display = 'none';

                const globalMonitor = document.getElementById('global-task-monitor');
                if (globalMonitor) globalMonitor.style.display = 'none';
                state.activeBatchTaskId = null;  // Clear active task
            }
        } catch (e) {
            console.error('Error polling task, stopping:', e);
            // Don't reschedule on error — stops the infinite retry loop
            if (progressBar) progressBar.style.display = 'none';
            if (progressText) progressText.style.display = 'none';
            state.activeBatchTaskId = null;
        }
    };

    setTimeout(checkProgress, 2000);
}


function setTool(tool) {
    state.currentTool = tool;
    state.currentPolygon = null;
    updateToolbar(); // Refresh button states
}

function setCurrentClass(classId) {
    state.currentClassId = parseInt(classId);
    hideClassSelectionPopup();
}

function showAddClassModal() {
    document.getElementById('add-class-modal').style.display = 'block';
    document.getElementById('new-class-name').focus();
    hideClassSelectionPopup();
}

function hideAddClassModal() {
    document.getElementById('add-class-modal').style.display = 'none';
    document.getElementById('new-class-name').value = '';
    document.getElementById('new-class-prompt').value = '';
}

async function submitNewClass() {
    const nameInput = document.getElementById('new-class-name');
    const colorInput = document.getElementById('new-class-color');
    const promptInput = document.getElementById('new-class-prompt');
    const name = nameInput.value.trim();
    const color = colorInput.value;
    const prompt = promptInput.value.trim();

    if (!name) {
        alert('Please enter a class name');
        return;
    }

    try {
        const updatedSchema = await apiRequest(`/projects/${state.currentProject.id}/classes`, {
            method: 'PATCH',
            body: JSON.stringify({ name, color, prompt })
        });

        state.classes = updatedSchema.classes;
        state.currentClassId = state.classes.length - 1;

        updateToolbar();
        hideAddClassModal();

        // If we were in the middle of a class selection for an annotation, re-show popup
        if (state.pendingAnnotation) {
            showClassSelectionPopup(state.pendingAnnotation, state.lastScreenPos);
        }

    } catch (error) {
        console.error('Failed to add class:', error);
    }
}

function addClassWhileAnnotating() {
    showAddClassModal();
}

// --- Edit Class Functions ---
function showEditClassModal(classId) {
    const cls = state.classes.find(c => c.id === classId);
    if (!cls) {
        console.error('Class not found:', classId);
        return;
    }

    document.getElementById('edit-class-id').value = classId;
    document.getElementById('edit-class-name').value = cls.name;
    document.getElementById('edit-class-color').value = cls.color || '#3498db';
    document.getElementById('edit-class-prompt').value = cls.prompt || '';
    document.getElementById('edit-class-modal').style.display = 'block';
}

function hideEditClassModal() {
    document.getElementById('edit-class-modal').style.display = 'none';
}

async function submitEditClass() {
    const classId = parseInt(document.getElementById('edit-class-id').value);
    const name = document.getElementById('edit-class-name').value.trim();
    const color = document.getElementById('edit-class-color').value;
    const prompt = document.getElementById('edit-class-prompt').value.trim();

    if (!name) {
        alert('Please enter a class name');
        return;
    }

    try {
        const updatedSchema = await apiRequest(`/projects/${state.currentProject.id}/classes/${classId}`, {
            method: 'PATCH',
            body: JSON.stringify({ name, color, prompt })
        });

        state.classes = updatedSchema.classes;
        updateToolbar();
        hideEditClassModal();
        showToast('Class updated successfully', 'success');

    } catch (error) {
        console.error('Failed to update class:', error);
        showToast('Failed to update class', 'error');
    }
}

async function deleteClassFromModal() {
    const classId = parseInt(document.getElementById('edit-class-id').value);
    const cls = state.classes.find(c => c.id === classId);

    if (!confirm(`Are you sure you want to delete the class "${cls?.name}"? This will NOT delete existing annotations, but they may become orphaned.`)) {
        return;
    }

    try {
        const updatedSchema = await apiRequest(`/projects/${state.currentProject.id}/classes/${classId}`, {
            method: 'DELETE'
        });

        state.classes = updatedSchema.classes;

        // Reset current class if it was deleted
        if (state.currentClassId === classId || !state.classes.find(c => c.id === state.currentClassId)) {
            state.currentClassId = state.classes.length > 0 ? state.classes[0].id : 0;
        }

        updateToolbar();
        hideEditClassModal();
        showToast('Class deleted successfully', 'success');

    } catch (error) {
        console.error('Failed to delete class:', error);
        showToast('Failed to delete class', 'error');
    }
}


function showClassSelectionPopup(annotation, screenPos) {
    state.pendingAnnotation = annotation;
    state.lastScreenPos = screenPos;

    const popup = document.getElementById('class-selection-popup');
    const list = document.getElementById('class-options-list');
    list.innerHTML = '';

    state.classes.forEach((cls, idx) => {
        const div = document.createElement('div');
        div.className = 'class-option';
        div.innerHTML = `
            <div class="class-option-color" style="background: ${cls.color || '#3498db'}"></div>
            <span>${cls.name}</span>
        `;
        div.onclick = () => {
            annotation.class_id = idx;
            hideClassSelectionPopup();
            pushToHistory();
            state.pendingAnnotation = null;
            drawImage();
        };
        list.appendChild(div);
    });

    popup.style.left = `${screenPos.x}px`;
    popup.style.top = `${screenPos.y}px`;
    popup.style.display = 'block';
}

function hideClassSelectionPopup() {
    document.getElementById('class-selection-popup').style.display = 'none';
}


function setCurrentKeypointIndex(index) {
    state.currentKeypointIndex = parseInt(index);
}

function clearAnnotations() {
    state.tempAnnotations = [];
    state.currentPolygon = null;
    state.selectedAnnotation = null;
    hideClassSelectionPopup();
    pushToHistory();
    drawImage();
    updateToolbar();
}

// Zoom functions
function zoomIn() {
    state.zoom *= 1.2;
    if (state.zoom > 10) state.zoom = 10;
    updateZoomDisplay();
    drawImage();
}

function zoomOut() {
    state.zoom /= 1.2;
    if (state.zoom < 0.1) state.zoom = 0.1;
    updateZoomDisplay();
    drawImage();
}

function resetZoom() {
    resetZoomAndFit();
}

function updateZoomDisplay() {
    document.getElementById('zoom-level').textContent = `${Math.round(state.zoom * 100)}%`;
}

// Modal functions
function showCreateProjectModal() {
    document.getElementById('create-project-modal').style.display = 'block';
}

function hideCreateProjectModal() {
    document.getElementById('create-project-modal').style.display = 'none';
}

function showImportModal() {
    document.getElementById('import-modal').style.display = 'block';
    showImageImport();
}

function hideImportModal() {
    document.getElementById('import-modal').style.display = 'none';
}

function showImageImport() {
    document.getElementById('import-image-section').style.display = 'block';
    document.getElementById('import-video-section').style.display = 'none';
}

function showVideoImport() {
    document.getElementById('import-image-section').style.display = 'none';
    document.getElementById('import-video-section').style.display = 'block';

    // Reset video import state
    document.getElementById('video-file-input').value = '';
    document.getElementById('target-fps').value = '1';
    document.getElementById('video-frame-preview').style.display = 'none';
}

// Video Import Preview Logic
// Video Import Preview Logic
document.getElementById('video-file-input')?.addEventListener('change', function (e) {
    if (this.files && this.files[0]) {
        captureVideoSample(this.files[0]);
    }
});



// Form handlers
document.getElementById('create-project-form').onsubmit = async function (e) {
    e.preventDefault();

    const nameInput = document.getElementById('project-name');
    const typeInput = document.getElementById('annotation-type');

    if (!nameInput.value) {
        showToast('Please enter a project name', 'warning');
        return;
    }

    if (!typeInput.value) {
        showToast('Please select an annotation type', 'warning');
        return;
    }

    // Collect classes
    const classInputs = document.querySelectorAll('#classes-list .class-item input[type="text"]');
    const classColors = document.querySelectorAll('#classes-list .class-item input[type="color"]');
    const classes = [];
    classInputs.forEach((input, index) => {
        if (input.value.trim()) {
            classes.push({
                id: index,
                name: input.value.trim(),
                color: classColors[index].value
            });
        }
    });

    if (classes.length === 0) {
        showToast('At least one class is required', 'warning');
        return;
    }

    // Check for duplicate class names
    const classNames = classes.map(c => c.name.toLowerCase());
    const duplicates = classNames.filter((name, idx) => classNames.indexOf(name) !== idx);
    if (duplicates.length > 0) {
        const dupList = [...new Set(duplicates)].map(d => `"${d}"`).join(', ');
        showToast(`Duplicate class name(s): ${dupList}. Each class must have a unique name.`, 'error');
        return;
    }


    // Keypoints validation
    let keypoints = [];
    if (typeInput.value === 'keypoints') {
        const kpInputs = document.querySelectorAll('#keypoints-list .keypoint-item input[type="text"]');
        kpInputs.forEach((input, index) => {
            if (input.value.trim()) {
                keypoints.push({
                    id: index,
                    name: input.value.trim()
                });
            }
        });
        if (keypoints.length === 0) {
            showToast('At least one keypoint label is required', 'warning');
            return;
        }
    }

    const projectData = {
        name: nameInput.value,
        annotation_type: typeInput.value,
        label_schema: {
            classes: classes,
            keypoints: keypoints
        }
    };

    const submitBtn = e.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Creating...';

    try {
        const response = await apiRequest('/projects', {
            method: 'POST',
            body: JSON.stringify(projectData)
        });

        hideCreateProjectModal();
        showToast('Project created successfully!', 'success');

        // Refresh grid
        loadProjects();

    } catch (error) {
        console.error('Create project failed:', error);
        showToast('Failed to create project: ' + error.message, 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
    }
};

function updateLabelSchemaUI() {
    const annotationType = document.getElementById('annotation-type').value;
    const keypointsSection = document.getElementById('keypoints-section');

    if (annotationType === 'keypoints') {
        keypointsSection.style.display = 'block';
    } else {
        keypointsSection.style.display = 'none';
    }
}

function addClass() {
    const colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];
    const color = colors[document.querySelectorAll('#classes-list .class-item').length % colors.length];

    const classItem = document.createElement('div');
    classItem.className = 'class-item';
    classItem.innerHTML = `
        <input type="color" class="class-color" value="${color}">
        <input type="text" class="form-control" placeholder="Class name" style="flex: 1;">
        <button type="button" class="btn btn-secondary" onclick="removeClass(this)">×</button>
    `;

    document.getElementById('classes-list').appendChild(classItem);
}

function removeClass(button) {
    const classList = document.getElementById('classes-list');
    if (classList.children.length > 1) {
        button.closest('.class-item').remove();
    } else {
        alert('At least one class is required');
    }
}

function addKeypoint() {
    const keypointItem = document.createElement('div');
    keypointItem.className = 'keypoint-item';
    keypointItem.style.display = 'flex';
    keypointItem.style.gap = '10px';
    keypointItem.style.marginBottom = '10px';
    keypointItem.style.alignItems = 'center';

    keypointItem.innerHTML = `
        <input type="text" class="form-control" placeholder="Keypoint name" style="flex: 1;">
        <button type="button" class="btn btn-secondary" onclick="this.closest('.keypoint-item').remove()">×</button>
    `;

    document.getElementById('keypoints-list').appendChild(keypointItem);
}

// Event Listeners
function setupEventListeners() {
    // Close modals when clicking outside
    window.addEventListener('click', (e) => {
        const createModal = document.getElementById('create-project-modal');
        const importModal = document.getElementById('import-modal');

        if (e.target === createModal) {
            hideCreateProjectModal();
        }
        if (e.target === importModal) {
            hideImportModal();
        }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Prevent shortcuts when typing in input fields
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
            return;
        }

        // Undo / Redo
        if (e.ctrlKey && e.key === 'z') { e.preventDefault(); undoAnnotations(); return; }
        if (e.ctrlKey && (e.key === 'y' || (e.shiftKey && e.key === 'Z'))) { e.preventDefault(); redoAnnotations(); return; }

        // Save/Verify with Ctrl+S, S, V, or Space
        if ((e.ctrlKey && (e.key === 's' || e.key === 'S')) ||
            ['s', 'S', 'v', 'V', ' '].includes(e.key)) {
            e.preventDefault();
            if (state.currentImage) {
                saveAnnotations();
            }
        }

        // Next image: Right Arrow, N, or D
        if (e.key === 'ArrowRight' || e.key === 'n' || e.key === 'N' || (e.key === 'd' && !e.ctrlKey)) {
            e.preventDefault();
            navigateToNextImage();
        }

        // Previous image: Left Arrow, P, or A
        if (e.key === 'ArrowLeft' || e.key === 'p' || e.key === 'P' || (e.key === 'a' && !e.ctrlKey)) {
            e.preventDefault();
            navigateToPreviousImage();
        }

        // Delete selected annotation or clear all
        if (e.key === 'Delete' || e.key === 'Backspace') {
            e.preventDefault();
            if (state.selectedAnnotation) {
                deleteSelectedAnnotation();
            } else {
                clearAnnotations();
            }
        }

        // Duplicate selected annotation with Ctrl+D
        if (e.ctrlKey && e.key === 'd') {
            e.preventDefault();
            duplicateSelectedAnnotation();
        }

        // Zoom with +/-
        if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomIn(); }
        if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomOut(); }
        if (e.key === '0') { e.preventDefault(); resetZoom(); }

        // Fit image to screen with F
        if (e.key === 'f' || e.key === 'F') {
            e.preventDefault();
            resetZoomAndFit();
        }

        // Toggle annotation visibility with H
        if (e.key === 'h' || e.key === 'H') {
            e.preventDefault();
            toggleAllAnnotationsVisibility();
        }

        // Cycle through annotations with Tab
        if (e.key === 'Tab') {
            e.preventDefault();
            cycleAnnotationSelection();
        }

        // Class selection with keys 1-9
        if (!e.ctrlKey && !e.altKey && e.key >= '1' && e.key <= '9') {
            const idx = parseInt(e.key) - 1;
            if (state.classes && state.classes[idx]) {
                setCurrentClass(state.classes[idx].id);
                showToast(`Class: ${state.classes[idx].name}`, 'info');
            }
        }

        // Show help with ?
        if (e.key === '?') {
            e.preventDefault();
            showKeyboardShortcutsHelp();
        }

        // Escape to cancel current drawing or deselect
        if (e.key === 'Escape') {
            e.preventDefault();
            if (state.selectedAnnotation) {
                deselectAnnotation();
            } else {
                cancelCurrentDrawing();
            }
        }
    });


    // Resize canvas on window resize
    window.addEventListener('resize', () => {
        resizeCanvas();
        // If we are in workspace, ensure the image fits
        if (state.currentImage && document.getElementById('workspace-view').style.display !== 'none') {
            resetZoomAndFit();
        }
    });
}

// New file browser upload functions
async function importImagesFromBrowser() {
    const imageInput = document.getElementById('image-file-input');
    const folderInput = document.getElementById('folder-input');

    let files = [];

    // Check which input has files
    if (imageInput.files.length > 0) {
        files = Array.from(imageInput.files);
    } else if (folderInput.files.length > 0) {
        files = Array.from(folderInput.files);
    }

    if (files.length === 0) {
        showToast('Please select images or a folder', 'warning');
        return;
    }

    try {
        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });

        const response = await fetch(`${API_BASE}/projects/${state.currentProject.id}/upload-images`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        const result = await response.json();

        hideImportModal();
        loadProjectImages(state.currentProject.id);
        showToast(`Import completed! ${result.imported_count} images imported.`, 'success');

        // Clear file inputs
        imageInput.value = '';
        folderInput.value = '';

    } catch (error) {
        console.error('Upload error:', error);
        showToast('Failed to upload images', 'error');
    }
}

async function importVideoFromBrowser() {
    const videoInput = document.getElementById('video-file-input');
    const targetFps = parseFloat(document.getElementById('target-fps').value);

    if (videoInput.files.length === 0) {
        alert('Please select a video file');
        return;
    }

    if (!targetFps || targetFps <= 0) {
        alert('Please enter a valid FPS value');
        return;
    }

    try {
        const formData = new FormData();
        formData.append('file', videoInput.files[0]);
        formData.append('fps', targetFps);

        // Show progress UI
        document.getElementById('import-progress-container').style.display = 'block';
        document.getElementById('import-progress-bar').style.width = '0%';
        document.getElementById('import-progress-text').textContent = '0%';
        const btn = document.querySelector('#import-video-section button');
        btn.disabled = true;

        const response = await fetch(`${API_BASE}/projects/${state.currentProject.id}/upload-video`, {
            method: 'POST',
            headers: {
                'Authorization': state.token ? `Bearer ${state.token}` : ''
            },
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        const result = await response.json();
        const taskId = result.task_id;

        // Poll for progress
        const pollInterval = setInterval(async () => {
            try {
                const statusRes = await fetch(`${API_BASE}/tasks/${taskId}`);
                if (!statusRes.ok) {
                    clearInterval(pollInterval);
                    if (statusRes.status === 404) {
                        console.warn('Task not found (server may have reloaded), stopping poll.');
                    } else {
                        console.error(`Task poll failed with status ${statusRes.status}, stopping poll.`);
                        showToast(`Video processing error (${statusRes.status})`, 'error');
                    }
                    btn.disabled = false;
                    document.getElementById('import-progress-container').style.display = 'none';
                    return;
                }
                const status = await statusRes.json();

                if (status.status === 'completed') {
                    clearInterval(pollInterval);
                    document.getElementById('import-progress-bar').style.width = '100%';
                    document.getElementById('import-progress-text').textContent = '100%';

                    setTimeout(() => {
                        hideImportModal();
                        loadProjectImages(state.currentProject.id);
                        showToast(`Extraction Complete! ${status.current} frames added.`, 'success');

                        // Reset UI
                        document.getElementById('import-progress-container').style.display = 'none';
                        btn.disabled = false;
                        videoInput.value = '';
                    }, 500);
                } else if (status.status === 'failed') {
                    clearInterval(pollInterval);
                    showToast(`Extraction failed: ${status.error}`, 'error');
                    btn.disabled = false;
                } else {
                    // Update progress
                    document.getElementById('import-progress-bar').style.width = `${status.progress}%`;
                    document.getElementById('import-progress-text').textContent = `${status.progress}% (${status.current}/${status.total})`;
                }
            } catch (e) {
                clearInterval(pollInterval);
                console.error('Polling error, stopped:', e);
                btn.disabled = false;
            }
        }, 1500);

    } catch (error) {
        console.error('Upload error:', error);
        showToast('Failed to upload video', 'error');
        document.getElementById('import-progress-container').style.display = 'none';
    }
}

// Helper to capture sample frame from video input
function captureVideoSample(file) {
    const video = document.getElementById('preview-video-hidden');
    const canvas = document.getElementById('preview-canvas');
    const durationSpan = document.getElementById('video-duration');
    const framesSpan = document.getElementById('estimated-frames');
    const previewDiv = document.getElementById('video-frame-preview');

    if (!video || !canvas) return;

    const url = URL.createObjectURL(file);
    video.src = url;

    video.onloadedmetadata = () => {
        // Show preview container
        previewDiv.style.display = 'block';

        // Seek to 10% or 1s to get a good frame
        video.currentTime = Math.min(1.0, video.duration * 0.1);

        // Update stats
        durationSpan.textContent = `${video.duration.toFixed(1)}s`;
        updateEstimatedFrames(); // Initial calculation
    };

    video.onseeked = () => {
        // Draw frame to canvas
        canvas.width = 300; // Fixed width for preview
        canvas.height = (video.videoHeight / video.videoWidth) * 300;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    };

    // Update frames estimation when FPS changes
    const fpsInput = document.getElementById('target-fps');
    fpsInput.oninput = updateEstimatedFrames;

    function updateEstimatedFrames() {
        const fps = parseFloat(fpsInput.value) || 1;
        const total = Math.floor(video.duration * fps);
        framesSpan.textContent = `${total} frames`;
    }
}

// Toast Notification System
// Utility: Shuffle array using Fisher-Yates algorithm
function shuffleArray(array) {
    const shuffled = [...array]; // Create a copy
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
}

function showToast(message, type = 'info', duration = 3000) {
    // Add to logs array
    const logEntry = {
        message: message,
        type: type,
        timestamp: new Date().toLocaleTimeString(),
        id: Date.now() + Math.random()
    };
    state.logs.push(logEntry);

    // Limit logs to last 100 entries
    if (state.logs.length > 100) state.logs.shift();

    // Update log indicator if it exists
    const logBadge = document.getElementById('log-badge');
    if (logBadge) {
        logBadge.style.display = 'block';
        logBadge.textContent = state.logs.length;
    }

    // Only show toast if it's not a background "annotated" or "running" message
    // or if the user specifically wants regular toasts.
    // In this case, the user asked to ONLY show in log popup.
    // However, for critical errors or success, we might still want a toast?
    // User said: "remove the model training notification from all the pages... 
    // also the notification like annotaed , running custom model etc 
    // all are only visible inside a small log popup"

    const lowercaseMsg = message.toLowerCase();
    const isBackgroundLog = lowercaseMsg.includes('annotated') ||
        lowercaseMsg.includes('running') ||
        lowercaseMsg.includes('custom model') ||
        lowercaseMsg.includes('shuffle') ||
        lowercaseMsg.includes('saved') ||
        lowercaseMsg.includes('found') ||
        lowercaseMsg.includes('streaming ai') ||
        lowercaseMsg.includes('verified') ||
        lowercaseMsg.includes('deleted') ||
        lowercaseMsg.includes('complete') ||
        lowercaseMsg.includes('null') ||
        lowercaseMsg.includes('batch:');

    if (isBackgroundLog) {
        console.log('Log recorded:', message);
        return; // Don't show toast for these
    }

    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    // Icon based on type
    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '❌';
    if (type === 'warning') icon = '⚠️';

    toast.innerHTML = `
        <div style="display: flex; align-items: center;">
            <span style="font-size: 20px; margin-right: 12px;">${icon}</span>
            <span class="toast-message">${message}</span>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()">×</button>
    `;

    container.appendChild(toast);

    // Auto remove
    setTimeout(() => {
        toast.style.animation = 'fadeOut 0.3s ease-out forwards';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}


function showConfirmation(title, message, callback) {
    document.getElementById('confirm-title').textContent = title;
    document.getElementById('confirm-message').textContent = message;
    document.getElementById('confirmation-overlay').style.display = 'flex';

    // Direct assignment - simpler and less prone to cloning issues
    const btn = document.getElementById('confirm-btn-yes');

    // Remove old listeners by cloning
    const newBtn = btn.cloneNode(true);
    btn.parentNode.replaceChild(newBtn, btn);

    newBtn.onclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        closeConfirmation();
        callback();
    };
}

function closeConfirmation() {
    document.getElementById('confirmation-overlay').style.display = 'none';
}

// Delete Image Functionality
async function deleteImage() {
    if (!state.currentImage) {
        showToast('No image selected', 'warning');
        return;
    }

    showConfirmation(
        'Delete Image?',
        'This action cannot be undone. The image and its annotations will be permanently removed.',
        async () => {
            const imageId = state.currentImage.id;
            const originalImages = [...state.images]; // Backup
            const deletedImageIndex = state.currentImageIndex;

            try {
                // Optimistic Update: Remove from UI list immediately
                state.images = state.images.filter(img => img.id !== imageId);
                state.totalImages = Math.max(0, state.totalImages - 1);

                // Remove from DOM immediately (sidebar thumbnail)
                const card = document.getElementById(`image-card-${imageId}`);
                if (card) card.remove();

                // Determine next image to load
                if (state.images.length > 0) {
                    let nextIndex = deletedImageIndex;

                    // If we deleted the last image, go back one
                    if (nextIndex >= state.images.length) {
                        nextIndex = state.images.length - 1;
                    }

                    // CRITICAL FIX: loadImage expects an ID, not an index!
                    const nextImageId = state.images[nextIndex].id;
                    state.currentImageIndex = nextIndex;

                    // Use a flag to avoid saving annotations for the deleted image
                    state.isDirty = false;
                    await loadImage(nextImageId);

                } else {
                    // No images left
                    state.currentImage = null;
                    state.currentImageIndex = -1;
                    imageObj = null;
                    document.getElementById('image-list').innerHTML = '<div style="padding: 20px; text-align: center; color: #7f8c8d;">No images available</div>';
                    drawImage();
                    updateCounter();
                }

                // Show success immediately
                showToast('Image deleted', 'success');

                // Perform API call in background
                await apiRequest(`/images/${imageId}`, { method: 'DELETE' });

            } catch (error) {
                console.error('Failed to delete image:', error);

                // Revert state on error (full reload is safest)
                showToast('Failed to delete image (reloading)', 'error');
                loadProjectImages(state.currentProject.id);
            }
        }
    );
}

// Null Image Functionality
// Null Image Functionality - Toggle status between Null (Processed + Empty) and Pending (New)
async function markAsNull() {
    if (!state.currentImage) {
        showToast('No image selected', 'warning');
        return;
    }

    // A "Null" image is one that is processed but has NO annotations.
    // We check tempAnnotations (unsaved changes) to see if the user is currently looking at an empty state.
    const isCurrentlyNull = state.currentImage.status === 'processed' &&
        state.tempAnnotations.length === 0;

    try {
        if (isCurrentlyNull) {
            // TOGGLE OFF: Revert to pending (Unmark as Null)
            await apiRequest(`/images/${state.currentImage.id}/status`, {
                method: 'PATCH',
                body: JSON.stringify({ status: 'pending', clear_annotations: true })
            });

            state.currentImage.status = 'pending';
            state.currentImage.has_annotations = false;
            state.currentImage.latest_version = 0;
            showToast('Null status removed - Reverted to New', 'success');
        } else {
            // TOGGLE ON: Mark as null
            await apiRequest(`/images/${state.currentImage.id}/null`, {
                method: 'POST'
            });

            state.currentImage.status = 'processed';
            state.currentImage.has_annotations = true;
            state.currentImage.latest_version = (state.currentImage.latest_version || 0) + 1;
            showToast('Image marked as Null (Negative Sample)', 'success');
        }

        // --- UI Updates ---

        // 1. Update list item badge
        const card = document.getElementById(`image-card-${state.currentImage.id}`);
        if (card) {
            const badge = card.querySelector('.status-badge');
            if (badge) {
                if (state.currentImage.status === 'processed') {
                    badge.className = 'status-badge status-processed';
                    badge.textContent = `v${state.currentImage.latest_version}`;
                } else {
                    badge.className = 'status-badge status-pending';
                    badge.textContent = 'New';
                }
            }
        }

        // 2. Update memory state for current list
        if (state.images) {
            const imgInList = state.images.find(img => img.id === state.currentImage.id);
            if (imgInList) {
                imgInList.status = state.currentImage.status;
                imgInList.has_annotations = state.currentImage.has_annotations;
                imgInList.latest_version = state.currentImage.latest_version;
            }
        }

        // 3. Clear workspace annotations if we just unmarked or marked as null
        state.tempAnnotations = [];
        state.currentAnnotations = [];
        state.isDirty = false;

        drawImage(); // Refresh watermark and objects

    } catch (error) {
        console.error('Failed to toggle null status:', error);
        showToast('Operation failed', 'error');
    }
}



// Image navigation
let currentImageList = [];
let currentImageIndex = -1;

async function navigateToNextImage() {
    if (!state.currentProject) {
        console.log('No project loaded');
        return;
    }

    // Auto-save before navigation
    await saveAnnotationsIfNeeded();

    // Get image list from the sidebar
    const imageElements = document.querySelectorAll('#image-list .project-card');
    if (imageElements.length === 0) {
        console.log('No images in project');
        return;
    }

    // Find current image index
    let activeIndex = -1;
    for (let i = 0; i < imageElements.length; i++) {
        if (imageElements[i].classList.contains('active')) {
            activeIndex = i;
            break;
        }
    }

    // Navigate to next (with wraparound)
    const nextIndex = (activeIndex + 1) % imageElements.length;
    imageElements[nextIndex].click();
}

function toggleImageShuffle() {
    if (!state.currentProject) return;

    const projectId = state.currentProject.id;
    const key = `project_${projectId}_shuffle_images`;
    const currentlyEnabled = localStorage.getItem(key) === 'true';

    localStorage.setItem(key, (!currentlyEnabled).toString());

    // Visual feedback
    const btn = document.querySelector('button[onclick="toggleImageShuffle()"]');
    if (btn) {
        btn.classList.toggle('btn-primary', !currentlyEnabled);
        btn.classList.toggle('btn-secondary', currentlyEnabled);
    }

    showToast(currentlyEnabled ? 'Image shuffle disabled' : 'Image shuffle enabled', 'info');

    // Reload images to apply shuffle
    loadProjectImages(projectId);
}

function toggleVerificationMode() {
    if (!state.currentProject) return;

    state.showUnverifiedOnly = !state.showUnverifiedOnly;
    const btn = document.getElementById('verification-mode-btn');
    if (btn) {
        btn.classList.toggle('btn-primary', state.showUnverifiedOnly);
        btn.classList.toggle('btn-secondary', !state.showUnverifiedOnly);
    }

    showToast(state.showUnverifiedOnly ? 'Verification mode: Showing AI suggestions' : 'Showing all images', 'info');
    loadProjectImages(state.currentProject.id);
}

async function confirmAndNext() {
    // Mark current as human and save
    const success = await saveAnnotations(false);
    if (success) {
        // Optimization: if we were in verification mode and this image no longer matches, 
        // the list will shrink. But navigateToNextImage uses DOM list, so it's safe.
        navigateToNextImage();
    }
}

function showLogsModal() {
    const modal = document.getElementById('logs-modal');
    const logsList = document.getElementById('logs-list');

    if (!modal || !logsList) return;

    logsList.innerHTML = '';

    if (state.logs.length === 0) {
        logsList.innerHTML = '<div style="padding: 20px; text-align: center; color: #888;">No logs recorded yet.</div>';
    } else {
        // Show logs in reverse order (newest first)
        [...state.logs].reverse().forEach(log => {
            const div = document.createElement('div');
            div.className = `log-entry ${log.type}`;
            div.innerHTML = `
                <span class="log-time">${log.timestamp}</span>
                <span class="log-message">${log.message}</span>
            `;
            logsList.appendChild(div);
        });
    }

    modal.style.display = 'block';

    // Reset badge
    const logBadge = document.getElementById('log-badge');
    if (logBadge) logBadge.style.display = 'none';
}

function hideLogsModal() {
    document.getElementById('logs-modal').style.display = 'none';
}

async function navigateToPreviousImage() {
    if (!state.currentProject) {
        console.log('No project loaded');
        return;
    }

    // Auto-save before navigation
    await saveAnnotationsIfNeeded();

    // Get image list from the sidebar
    const imageElements = document.querySelectorAll('#image-list .project-card');
    if (imageElements.length === 0) {
        console.log('No images in project');
        return;
    }

    // Find current image index
    let activeIndex = -1;
    for (let i = 0; i < imageElements.length; i++) {
        if (imageElements[i].classList.contains('active')) {
            activeIndex = i;
            break;
        }
    }

    // Navigate to previous (with wraparound)
    const prevIndex = activeIndex <= 0 ? imageElements.length - 1 : activeIndex - 1;
    imageElements[prevIndex].click();
}

function cancelCurrentDrawing() {
    if (state.currentPolygon) {
        state.currentPolygon = null;
        state.tempAnnotations = state.tempAnnotations.filter(a => !a.drawing);
        drawImage();
        console.log('Cancelled current drawing');
    }
}

function showKeyboardShortcutsHelp() {
    showHelpView();
    switchHelpTab('shortcuts');
}

// ===========================================
// ANNOTATION SELECTION AND EDITING FUNCTIONS
// ===========================================

// Toggle between draw and select modes
function setMode(mode) {
    state.selectionMode = mode;
    state.selectedAnnotation = null;
    state.isDragging = false;
    state.isResizing = false;

    // Update cursor
    if (mode === 'select') {
        canvas.style.cursor = 'default';
    } else {
        canvas.style.cursor = 'crosshair';
    }

    updateToolbar();
    drawImage();
}

// Check if a point is inside a bounding box
function hitTestBbox(pos, bbox) {
    return pos.x >= bbox.x &&
        pos.x <= bbox.x + bbox.width &&
        pos.y >= bbox.y &&
        pos.y <= bbox.y + bbox.height;
}

// Check if a point is inside a polygon
function hitTestPolygon(pos, polygon) {
    if (!polygon.points || polygon.points.length < 3) return false;

    let inside = false;
    const points = polygon.points;

    for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
        const xi = points[i].x, yi = points[i].y;
        const xj = points[j].x, yj = points[j].y;

        if (((yi > pos.y) !== (yj > pos.y)) &&
            (pos.x < (xj - xi) * (pos.y - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }

    return inside;
}

// Hit test all annotations and return the one under the cursor
function hitTestAnnotation(pos, currentSelected = null) {
    // Collect all annotations that hit this position
    const hits = [];

    // Check in reverse order (last drawn = on top)
    for (let i = state.tempAnnotations.length - 1; i >= 0; i--) {
        const ann = state.tempAnnotations[i];

        let isHit = false;
        if (ann.type === 'bbox' || state.annotationType === 'bbox') {
            if (hitTestBbox(pos, ann)) isHit = true;
        } else if (ann.type === 'polygon' || state.annotationType === 'polygon') {
            if (hitTestPolygon(pos, ann)) isHit = true;
        }

        if (isHit) hits.push(ann);
    }

    if (hits.length === 0) return null;

    // Selection Cycling: If current selection is one of the hits, pick the next one after it
    if (currentSelected && hits.includes(currentSelected)) {
        const currentIndex = hits.indexOf(currentSelected);
        const nextIndex = (currentIndex + 1) % hits.length;
        return hits[nextIndex];
    }

    // Default: return the top-most (first in hits)
    return hits[0];
}

// Get resize handles for a bbox annotation
function getBboxResizeHandles(bbox) {
    const handleSize = 8 / state.zoom;
    return [
        { id: 'nw', x: bbox.x - handleSize / 2, y: bbox.y - handleSize / 2 },
        { id: 'ne', x: bbox.x + bbox.width - handleSize / 2, y: bbox.y - handleSize / 2 },
        { id: 'sw', x: bbox.x - handleSize / 2, y: bbox.y + bbox.height - handleSize / 2 },
        { id: 'se', x: bbox.x + bbox.width - handleSize / 2, y: bbox.y + bbox.height - handleSize / 2 },
        { id: 'n', x: bbox.x + bbox.width / 2 - handleSize / 2, y: bbox.y - handleSize / 2 },
        { id: 's', x: bbox.x + bbox.width / 2 - handleSize / 2, y: bbox.y + bbox.height - handleSize / 2 },
        { id: 'w', x: bbox.x - handleSize / 2, y: bbox.y + bbox.height / 2 - handleSize / 2 },
        { id: 'e', x: bbox.x + bbox.width - handleSize / 2, y: bbox.y + bbox.height / 2 - handleSize / 2 }
    ];
}

// Check if point is on a resize handle
function hitTestResizeHandles(pos, annotation) {
    if (!annotation || annotation.type !== 'bbox') return null;

    const handles = getBboxResizeHandles(annotation);
    const handleSize = 8 / state.zoom;

    for (const handle of handles) {
        if (pos.x >= handle.x && pos.x <= handle.x + handleSize &&
            pos.y >= handle.y && pos.y <= handle.y + handleSize) {
            return handle.id;
        }
    }
    return null;
}

// Select an annotation
function selectAnnotation(annotation, screenPos = null) {
    hideClassSelectionPopup();
    state.selectedAnnotation = annotation;
    drawImage();
    updateToolbar();

    // If screen position provided (from click), show the class switcher popup there
    if (screenPos) {
        showClassSelectionPopup(annotation, screenPos);
    }
}

// Deselect current annotation
function deselectAnnotation() {
    state.selectedAnnotation = null;
    state.isDragging = false;
    state.isResizing = false;
    drawImage();
    updateToolbar();
    hideClassSelectionPopup();
}

// Delete currently selected annotation
function deleteSelectedAnnotation() {
    if (state.selectedAnnotation) {
        state.tempAnnotations = state.tempAnnotations.filter(a => a !== state.selectedAnnotation);
        state.selectedAnnotation = null;
        hideClassSelectionPopup();
        pushToHistory();
        drawImage();
        renderInstancePanel();
        updateToolbar();
    }
}

// Duplicate the currently selected annotation (Ctrl+D)
function duplicateSelectedAnnotation() {
    if (!state.selectedAnnotation) return;
    const clone = JSON.parse(JSON.stringify(state.selectedAnnotation));
    // Offset slightly so it's visible
    if (clone.type === 'bbox') {
        clone.x = Math.min(clone.x + 0.02, 1 - clone.width);
        clone.y = Math.min(clone.y + 0.02, 1 - clone.height);
    } else if (clone.type === 'polygon' && clone.points) {
        clone.points = clone.points.map(p => ({ x: Math.min(p.x + 0.02, 1), y: Math.min(p.y + 0.02, 1) }));
    }
    state.tempAnnotations.push(clone);
    state.selectedAnnotation = clone;
    pushToHistory();
    drawImage();
    renderInstancePanel();
    showToast('Annotation duplicated', 'success');
}

// Toggle visibility of all annotations (H key)
let _annotationsHidden = false;
function toggleAllAnnotationsVisibility() {
    _annotationsHidden = !_annotationsHidden;
    state.tempAnnotations.forEach(a => { a._hidden = _annotationsHidden; });
    drawImage();
    renderInstancePanel();
    showToast(_annotationsHidden ? '👁 Annotations hidden' : '👁 Annotations visible', 'info');
}

// Cycle through annotations with Tab
function cycleAnnotationSelection() {
    if (!state.tempAnnotations || state.tempAnnotations.length === 0) return;
    const visible = state.tempAnnotations.filter(a => !a.drawing);
    if (visible.length === 0) return;
    const currentIdx = visible.indexOf(state.selectedAnnotation);
    const nextIdx = (currentIdx + 1) % visible.length;
    selectAnnotation(visible[nextIdx]);
    renderInstancePanel();
}

// =============================================
// INSTANCE PANEL
// =============================================
function renderInstancePanel() {
    const panel = document.getElementById('instance-panel-list');
    if (!panel) return;
    panel.innerHTML = '';

    const annotations = (state.tempAnnotations || []).filter(a => !a.drawing);
    const badge = document.getElementById('instance-count-badge');
    if (badge) badge.textContent = annotations.length;

    if (annotations.length === 0) {
        panel.innerHTML = '<div style="color:#aaa;padding:12px;font-size:13px;text-align:center;">No annotations yet</div>';
        return;
    }

    annotations.forEach((ann, idx) => {
        const color = getClassColor(ann.class_id);
        const name = getClassName(ann.class_id);
        const isSelected = ann === state.selectedAnnotation;
        const isHidden = ann._hidden;

        const item = document.createElement('div');
        item.className = 'instance-item';
        item.style.cssText = `
            display:flex; align-items:center; gap:8px; padding:8px 10px;
            background:${isSelected ? 'rgba(52,152,219,0.15)' : 'transparent'};
            border-left:3px solid ${isSelected ? '#3498db' : 'transparent'};
            border-radius:4px; cursor:pointer; transition:background 0.15s;
            margin-bottom:2px; opacity:${isHidden ? 0.4 : 1};
        `;

        item.innerHTML = `
            <div style="width:12px;height:12px;border-radius:3px;background:${color};flex-shrink:0;"></div>
            <div style="flex:1;min-width:0;">
                <div style="font-size:12px;font-weight:600;color:#2c3e50;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${name}</div>
                <div style="font-size:10px;color:#95a5a6;">${ann.type || 'annotation'} #${idx + 1}</div>
            </div>
            <button title="${isHidden ? 'Show' : 'Hide'}" onclick="event.stopPropagation(); toggleAnnotationVisibility(${idx})" 
                style="background:none;border:none;cursor:pointer;font-size:14px;padding:2px;opacity:0.6;">${isHidden ? '🙈' : '👁'}</button>
            <button title="Delete" onclick="event.stopPropagation(); deleteAnnotationByIndex(${idx})" 
                style="background:none;border:none;cursor:pointer;font-size:13px;padding:2px;color:#e74c3c;opacity:0.7;">✕</button>
        `;

        item.onclick = () => {
            selectAnnotation(ann);
            renderInstancePanel();
        };

        item.onmouseenter = () => { if (!isSelected) item.style.background = 'rgba(0,0,0,0.04)'; };
        item.onmouseleave = () => { if (!isSelected) item.style.background = 'transparent'; };

        panel.appendChild(item);
    });
}

function toggleAnnotationVisibility(idx) {
    const visible = state.tempAnnotations.filter(a => !a.drawing);
    if (visible[idx]) {
        visible[idx]._hidden = !visible[idx]._hidden;
        drawImage();
        renderInstancePanel();
    }
}

function deleteAnnotationByIndex(idx) {
    const visible = state.tempAnnotations.filter(a => !a.drawing);
    if (visible[idx]) {
        state.tempAnnotations = state.tempAnnotations.filter(a => a !== visible[idx]);
        if (state.selectedAnnotation === visible[idx]) state.selectedAnnotation = null;
        pushToHistory();
        drawImage();
        renderInstancePanel();
    }
}



// Draw selection handles on selected annotation
function drawSelectionHandles(annotation) {
    if (!annotation) return;

    ctx.save();

    if (annotation.type === 'bbox' || state.annotationType === 'bbox') {
        // Draw selection border
        ctx.strokeStyle = '#00AAFF';
        ctx.lineWidth = 2 / state.zoom;
        ctx.setLineDash([5 / state.zoom, 5 / state.zoom]);
        ctx.strokeRect(annotation.x, annotation.y, annotation.width, annotation.height);
        ctx.setLineDash([]);

        // Draw resize handles
        const handles = getBboxResizeHandles(annotation);
        const handleSize = 8 / state.zoom;

        ctx.fillStyle = '#FFFFFF';
        ctx.strokeStyle = '#00AAFF';
        ctx.lineWidth = 2 / state.zoom;

        handles.forEach(handle => {
            ctx.fillRect(handle.x, handle.y, handleSize, handleSize);
            ctx.strokeRect(handle.x, handle.y, handleSize, handleSize);
        });
    } else if (annotation.type === 'polygon' || state.annotationType === 'polygon') {
        // Draw selection highlight for polygon points
        const handleSize = 6 / state.zoom;

        ctx.fillStyle = '#00AAFF';
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2 / state.zoom;

        annotation.points.forEach(point => {
            ctx.beginPath();
            ctx.arc(point.x, point.y, handleSize, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        });

        // Draw ghost point if hovering over an edge
        if (state.hoveredEdgePoint && state.selectedAnnotation === annotation) {
            ctx.beginPath();
            ctx.arc(state.hoveredEdgePoint.x, state.hoveredEdgePoint.y, handleSize, 0, Math.PI * 2);
            ctx.fillStyle = '#FFFFFF';
            ctx.strokeStyle = '#00AAFF';
            ctx.fill();
            ctx.stroke();

            // Draw a small plus sign inside
            ctx.beginPath();
            ctx.moveTo(state.hoveredEdgePoint.x - 3, state.hoveredEdgePoint.y);
            ctx.lineTo(state.hoveredEdgePoint.x + 3, state.hoveredEdgePoint.y);
            ctx.moveTo(state.hoveredEdgePoint.x, state.hoveredEdgePoint.y - 3);
            ctx.lineTo(state.hoveredEdgePoint.x, state.hoveredEdgePoint.y + 3);
            ctx.strokeStyle = '#00AAFF';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    }

    ctx.restore();
}

// --- Export Logic ---

let exportState = {
    projectId: null,
    step: 1,
    config: {
        format: 'yolo',
        split_ratios: [0.8, 0.1, 0.1],
        resize: null,
        grayscale: false,
        augmentation: {
            enabled: false,
            brightness: 0,
            contrast: 0,
            blur: 0,
            noise: 0,
            rotation: 0,
            count_multiplier: 1
        }
    }
};

function showExportView() {
    document.getElementById('dashboard-view').style.display = 'none';
    document.getElementById('workspace-view').style.display = 'none';
    document.getElementById('manage-projects-view').style.display = 'none';
    document.getElementById('export-project-view').style.display = 'flex';
    loadExportProjects();
}

async function loadExportProjects() {
    try {
        const projects = await apiRequest('/projects');
        const grid = document.getElementById('export-projects-grid');
        grid.innerHTML = '';

        for (const project of projects) {
            const card = document.createElement('div');
            card.className = 'dashboard-card';
            card.style.alignItems = 'flex-start';
            card.style.textAlign = 'left';
            card.style.cursor = 'pointer';
            card.style.padding = '0';
            card.style.overflow = 'hidden';

            // Fetch first image for thumbnail
            let thumbnailHtml = `<div style="width: 100%; height: 150px; background: #f0f2f5; border-radius: 8px 8px 0 0; margin-bottom: 0; display: flex; align-items: center; justify-content: center; overflow: hidden; border-bottom: 1px solid #eee;">
                <span style="font-size: 3rem;">📁</span>
            </div>`;

            if (project.image_count > 0) {
                try {
                    const timestamp = new Date().getTime();
                    const imagesData = await apiRequest(`/projects/${project.id}/images?limit=1&_t=${timestamp}`);
                    if (imagesData.images && imagesData.images.length > 0) {
                        const img = imagesData.images[0];
                        thumbnailHtml = `<div style="width: 100%; height: 150px; background: #fff; border-radius: 8px 8px 0 0; margin-bottom: 0; overflow: hidden; border-bottom: 1px solid #eee;">
                            <img src="${API_BASE}/images/${img.id}/file?_t=${timestamp}" style="width: 100%; height: 100%; object-fit: cover;" alt="${project.name}">
                        </div>`;
                    }
                } catch (e) {
                    console.error('Failed to load thumbnail', e);
                }
            }

            card.innerHTML = `
                ${thumbnailHtml}
                <div style="padding: 15px; width: 100%;">
                    <h3 style="margin: 0 0 5px 0; font-size: 1.1rem; color: #2c3e50;">${project.name}</h3>
                    <p style="margin: 0 0 10px 0; font-size: 0.9rem; color: #7f8c8d;">${project.annotation_type}</p>
                    <p style="margin: 0; font-weight: 500;">
                        ${project.image_count} images
                    </p>
                </div>
            `;

            card.onclick = () => initExportWizard(project.id);
            grid.appendChild(card);
        }
    } catch (e) {
        console.error('Failed to load projects', e);
    }
}

function initExportWizard(projectId) {
    exportState.projectId = projectId;
    exportState.step = 1;
    exportState.config = {
        format: 'yolo',
        split_ratios: [0.8, 0.1, 0.1],
        resize: null,
        grayscale: false,
        augmentation: {
            enabled: false,
            brightness: 0,
            contrast: 0,
            blur: 0,
            noise: 0,
            rotation: 0,
            count_multiplier: 1
        }
    };

    // Reset UI
    document.querySelectorAll('.format-card').forEach(c => c.classList.remove('selected'));
    document.querySelector('.format-card:first-child').classList.add('selected'); // Default yolo
    document.getElementById('split-train').value = 80;
    document.getElementById('split-val').value = 10;
    document.getElementById('split-test').value = 10;
    document.getElementById('resize-w').value = '';
    document.getElementById('resize-h').value = '';
    document.getElementById('export-grayscale').checked = false;

    // Reset Augmentation UI
    document.getElementById('aug-brightness').value = 0;
    document.getElementById('val-brightness').innerText = '0';
    document.getElementById('aug-contrast').value = 0;
    document.getElementById('val-contrast').innerText = '0';
    document.getElementById('aug-blur').value = 0;
    document.getElementById('val-blur').innerText = '0';
    document.getElementById('aug-noise').value = 0;
    document.getElementById('val-noise').innerText = '0';
    document.getElementById('aug-rotation').value = 0;
    document.getElementById('val-rotation').innerText = '0';
    document.getElementById('export-multiplier').value = 1;
    document.getElementById('est-output-count').innerText = "...";

    updateWizardUI();
    document.getElementById('export-wizard-modal').style.display = 'flex';

    // Load initial preview (original image)
    loadExportPreviewImage();
}

function closeExportWizard() {
    document.getElementById('export-wizard-modal').style.display = 'none';
}

let previewDebounceTimer;
let currentPreviewImageId = null;

async function loadExportPreviewImage() {
    // Find a 'processed' image from the project to use as preview sample
    try {
        // Fetch project images directly
        // Default limit 100 should be enough to find a sample
        const response = await apiRequest(`/projects/${exportState.projectId}/images?limit=100`);
        const images = response.images;

        // Prefer processed images, but take any if needed
        const processedImages = images.filter(img => img.status === 'processed');
        const sample = processedImages.length > 0 ? processedImages[0] : (images.length > 0 ? images[0] : null);

        if (sample) {
            currentPreviewImageId = sample.id;

            // Set original image source
            document.getElementById('preview-original').src = `/api/images/${sample.id}/file`;
            document.getElementById('preview-aug').src = `/api/images/${sample.id}/file`; // Start identical

            // Update estimation baseline
            // Use response.total from the API for accurate count
            exportState.totalImages = response.total;
            updateDatasetEstimation();
        } else {
            console.warn("No images found for preview");
            exportState.totalImages = 0;
            document.getElementById('est-output-count').innerText = "No images found";
            // Set placeholders
            document.getElementById('preview-original').src = "";
            document.getElementById('preview-aug').src = "";
        }
    } catch (e) {
        console.error("Failed to load preview sample", e);
    }
}

function updatePreview() {
    // Update labels
    document.getElementById('val-brightness').innerText = document.getElementById('aug-brightness').value;
    document.getElementById('val-contrast').innerText = document.getElementById('aug-contrast').value;
    document.getElementById('val-blur').innerText = document.getElementById('aug-blur').value;
    document.getElementById('val-noise').innerText = document.getElementById('aug-noise').value;
    document.getElementById('val-rotation').innerText = document.getElementById('aug-rotation').value;

    if (!currentPreviewImageId) return;

    // Debounce API call
    clearTimeout(previewDebounceTimer);
    document.getElementById('preview-loading').style.display = 'block';

    previewDebounceTimer = setTimeout(async () => {
        const config = getAugmentationConfigFromUI();

        if (!config.enabled && !document.getElementById('export-grayscale').checked) {
            // Just show original if nothing enabled (technically should show resize/grayscale but preview endpoint handles separate logic?)
            // Preview endpoint handles aug config.
            // If we want to preview grayscale/resize, we should probably include them in preview endpoint?
            // Current preview endpoint only takes AugmentationConfig.
            // Let's assume preview is mostly for Augmentation.
            // But Grayscale is 'Preprocessing'.
            // Ideally preview checks everything.
            // For now, let's just send the augmentation config.
        }

        try {
            const isGrayscale = document.getElementById('export-grayscale').checked;
            const response = await apiRequest('/preview/augment', {
                method: 'POST',
                body: JSON.stringify({
                    image_id: currentPreviewImageId,
                    augmentation: config,
                    grayscale: isGrayscale
                })
            });
            document.getElementById('preview-aug').src = response.image;
        } catch (e) {
            console.error("Preview failed", e);
        } finally {
            document.getElementById('preview-loading').style.display = 'none';
        }
    }, 500); // 500ms debounce
}

function getAugmentationConfigFromUI() {
    const brightness = parseFloat(document.getElementById('aug-brightness').value);
    const contrast = parseFloat(document.getElementById('aug-contrast').value);
    const blur = parseFloat(document.getElementById('aug-blur').value);
    const noise = parseFloat(document.getElementById('aug-noise').value);
    const rotation = parseFloat(document.getElementById('aug-rotation').value);

    const enabled = (brightness !== 0 || contrast !== 0 || blur !== 0 || noise !== 0 || rotation !== 0);

    return {
        enabled: enabled,
        brightness: brightness,
        contrast: contrast,
        blur: blur,
        noise: noise,
        rotation: rotation,
        count_multiplier: parseInt(document.getElementById('export-multiplier').value)
    };
}

function updateDatasetEstimation() {
    if (!exportState.totalImages) return;
    const mult = parseInt(document.getElementById('export-multiplier').value);
    const total = exportState.totalImages * mult;
    document.getElementById('est-output-count').innerText = `${total} images`;
}

function selectExportFormat(card, format) {
    document.querySelectorAll('.format-card').forEach(c => c.classList.remove('selected'));
    card.classList.add('selected');
    exportState.config.format = format;
}

function exportWizardNext() {
    if (exportState.step === 2) {
        // Validate splits
        const train = parseInt(document.getElementById('split-train').value) || 0;
        const val = parseInt(document.getElementById('split-val').value) || 0;
        const test = parseInt(document.getElementById('split-test').value) || 0;

        if (train + val + test !== 100) {
            document.getElementById('split-error').style.display = 'block';
            return;
        }
        document.getElementById('split-error').style.display = 'none';
        exportState.config.split_ratios = [train / 100, val / 100, test / 100];
    }

    exportState.step++;
    updateWizardUI();
}

function exportWizardBack() {
    exportState.step--;
    updateWizardUI();
}

function updateWizardUI() {
    // Hide all steps
    document.querySelectorAll('.wizard-step').forEach(s => s.style.display = 'none');
    document.getElementById(`export-step-${exportState.step}`).style.display = 'block';

    const backBtn = document.getElementById('export-btn-back');
    const nextBtn = document.getElementById('export-btn-next');
    const finishBtn = document.getElementById('export-btn-finish');

    if (exportState.step === 1) {
        backBtn.style.display = 'none';
        nextBtn.style.display = 'block';
        finishBtn.style.display = 'none';
    } else if (exportState.step === 2) {
        backBtn.style.display = 'block';
        nextBtn.style.display = 'block';
        finishBtn.style.display = 'none';

        // Show test split only if needed? For now always show 3
    } else if (exportState.step === 3) {
        backBtn.style.display = 'block';
        nextBtn.style.display = 'none';
        finishBtn.style.display = 'block';
    }
}

async function startExport() {
    const w = parseInt(document.getElementById('resize-w').value);
    const h = parseInt(document.getElementById('resize-h').value);
    if (w && h) {
        exportState.config.resize = [w, h];
    }
    exportState.config.grayscale = document.getElementById('export-grayscale').checked;
    exportState.config.augmentation = getAugmentationConfigFromUI();

    // Close modal
    document.getElementById('export-wizard-modal').style.display = 'none';

    // Show loading toast
    showToast('Starting export...', 'info');

    try {
        const response = await apiRequest(`/projects/${exportState.projectId}/export`, {
            method: 'POST',
            body: JSON.stringify(exportState.config)
        });

        pollExportTask(response.task_id);
    } catch (e) {
        console.error(e);
        showToast('Export failed: ' + e.message, 'error');
    }
}

async function pollExportTask(taskId) {
    // Show the global task monitor for export progress
    const monitor = document.getElementById('global-task-monitor');
    const label = document.getElementById('global-task-label');
    const percent = document.getElementById('global-task-percent');
    const fill = document.getElementById('global-task-fill');
    const icon = document.getElementById('global-task-icon');

    monitor.style.display = 'flex';
    icon.textContent = '📦';
    label.textContent = 'EXPORTING DATASET...';
    percent.textContent = '0%';
    fill.style.width = '0%';

    const pollInterval = setInterval(async () => {
        try {
            const task = await apiRequest(`/tasks/${taskId}`, { method: 'GET' });
            const progress = task.progress || 0;

            if (task.status === 'zipping') {
                label.textContent = 'COMPRESSING ZIP...';
                icon.textContent = '🗜️';
                percent.textContent = `${progress}%`;
                fill.style.width = `${progress}%`;
            } else if (task.status === 'processing') {
                label.textContent = task.message || 'EXPORTING DATASET...';
                percent.textContent = `${progress}%`;
                fill.style.width = `${progress}%`;
            } else if (task.status === 'completed') {
                clearInterval(pollInterval);
                // Start download with progress tracking
                label.textContent = 'DOWNLOADING ZIP...';
                icon.textContent = '⬇️';
                percent.textContent = '0%';
                fill.style.width = '0%';
                fill.style.background = 'linear-gradient(90deg, #3498db 0%, #2ecc71 100%)';

                try {
                    await downloadWithProgress(taskId, task.zip_size || 0, label, percent, fill);
                    label.textContent = 'DOWNLOAD COMPLETE!';
                    icon.textContent = '✅';
                    percent.textContent = '100%';
                    fill.style.width = '100%';
                    showToast('Export downloaded successfully!', 'success');
                    setTimeout(() => { monitor.style.display = 'none'; }, 3000);
                } catch (dlErr) {
                    console.error('Download error:', dlErr);
                    showToast('Download failed. Retrying with direct link...', 'warning');
                    // Fallback to simple anchor download
                    const a = document.createElement('a');
                    a.href = `${window.location.origin}/api/exports/download/${taskId}`;
                    a.download = '';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    setTimeout(() => { monitor.style.display = 'none'; }, 2000);
                }
            } else if (task.status === 'failed') {
                clearInterval(pollInterval);
                monitor.style.display = 'none';
                showToast(`Export Failed: ${task.error}`, 'error');
            }
        } catch (e) {
            clearInterval(pollInterval);
            monitor.style.display = 'none';
            showToast('Lost connection to export task.', 'error');
        }
    }, 1000);
}

async function downloadWithProgress(taskId, totalSize, label, percent, fill) {
    const url = `${window.location.origin}/api/exports/download/${taskId}`;
    const response = await fetch(url);

    if (!response.ok) throw new Error(`Download failed: ${response.status}`);

    // Try to get content-length from response headers (more reliable than task.zip_size)
    const contentLength = response.headers.get('content-length');
    const total = contentLength ? parseInt(contentLength, 10) : totalSize;

    if (!response.body || !total) {
        // ReadableStream not supported or unknown size — fallback to blob download
        const blob = await response.blob();
        triggerBlobDownload(blob, 'dataset_export.zip');
        return;
    }

    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;

        const pct = Math.min(Math.round((received / total) * 100), 100);
        const sizeMB = (received / (1024 * 1024)).toFixed(1);
        const totalMB = (total / (1024 * 1024)).toFixed(1);

        percent.textContent = `${pct}%`;
        fill.style.width = `${pct}%`;
        label.textContent = `DOWNLOADING... ${sizeMB} / ${totalMB} MB`;
    }

    // Combine chunks and trigger download
    const blob = new Blob(chunks, { type: 'application/zip' });
    triggerBlobDownload(blob, 'dataset_export.zip');
}

function triggerBlobDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// --- Training Functions ---
function showTrainingModal() {
    if (!state.currentProject) return;

    const modal = document.getElementById('training-modal');
    const stats = document.getElementById('training-stats');
    const startBtn = document.getElementById('start-train-btn');

    // Check how many verified images we have
    const verifiedCount = state.images.filter(img => img.status === 'processed').length;

    stats.innerHTML = `
        <strong>Dataset Stats:</strong><br>
        • Total Images: ${state.images.length}<br>
        • Ground Truth (Verified): ${verifiedCount}<br>
        • Status: ${verifiedCount < 5 ? `<span style="color: #e74c3c;">⚠️ Need at least 5 to start</span>` : `<span style="color: #27ae60;">✅ Ready to train</span>`}
    `;

    startBtn.disabled = verifiedCount < 5;
    modal.style.display = 'flex';
}

function hideTrainingModal() {
    document.getElementById('training-modal').style.display = 'none';
}

async function startTraining() {
    const epochsInput = document.getElementById('train-epochs').value.trim();
    const multiplier = document.getElementById('train-multiplier').value;

    // Validate epoch input
    const epochs = parseInt(epochsInput);
    if (isNaN(epochs) || epochs < 1 || epochs > 1000) {
        showToast('❌ Please enter a valid epoch value (1-1000)', 'error');
        return;
    }

    showToast('🚀 Starting Trainer...', 'info');
    hideTrainingModal();

    try {
        const result = await apiRequest(`/projects/${state.currentProject.id}/train`, {
            method: 'POST',
            body: JSON.stringify({
                epochs: epochs,
                augment_multiplier: parseInt(multiplier)
            })
        });

        showToast('Training started in background!', 'success');
        if (result.task_id) {
            state.activeTrainingTaskId = result.task_id;
            localStorage.setItem('activeTrainingTaskId', result.task_id);
            pollTrainingProgress(result.task_id);
        }
    } catch (error) {
        console.error('Training failed:', error);
        showToast('Failed to start training', 'error');
    }
}

function pollTrainingProgress(taskId) {
    const progressBar = document.getElementById('ai-progress-toolbar');
    const progressFill = document.getElementById('ai-progress-fill-toolbar');
    const progressText = document.getElementById('ai-progress-text-toolbar');

    // Global monitor elements
    const globalMonitor = document.getElementById('global-task-monitor');
    const globalFill = document.getElementById('global-task-fill');
    const globalPercent = document.getElementById('global-task-percent');

    // Store in global state and localStorage for persistence
    state.activeTrainingTaskId = taskId;
    localStorage.setItem('activeTrainingTaskId', taskId);

    const checkStatus = async () => {
        try {
            const task = await apiRequest(`/tasks/${taskId}`);

            const isActive = ['augmenting', 'training', 'pending', 'queued', 'processing'].includes(task.status);

            if (isActive) {
                // Update Workspace Toolbar (if visible)
                if (progressBar) progressBar.style.display = 'flex';
                if (progressFill) progressFill.style.width = `${task.progress}%`;
                if (progressText) {
                    progressText.style.display = 'inline-block';
                    if (task.epoch && task.total_epochs) {
                        progressText.textContent = `🎓 Training: ${task.epoch}/${task.total_epochs} Epochs (${task.progress}%)`;
                    } else if (task.status === 'pending' || task.status === 'queued') {
                        progressText.textContent = `🎓 Training: Queued...`;
                    } else {
                        progressText.textContent = `🎓 Training: ${task.progress}%`;
                    }
                }

                // Update Global Monitor (visible across all views)
                if (globalMonitor) {
                    globalMonitor.style.display = 'flex';
                    if (globalFill) globalFill.style.width = `${task.progress}%`;
                    if (globalPercent) globalPercent.textContent = `${task.progress}%`;
                    const globalLabel = document.getElementById('global-task-label');
                    const globalIcon = document.getElementById('global-task-icon');
                    if (globalLabel) {
                        if (task.epoch && task.total_epochs) {
                            globalLabel.textContent = `TRAINING: EPOCH ${task.epoch}/${task.total_epochs}`;
                        } else if (task.status === 'pending' || task.status === 'queued') {
                            globalLabel.textContent = 'TRAINING: QUEUED...';
                        } else {
                            globalLabel.textContent = (task.status || 'PROCESSING...').toUpperCase();
                        }
                    }
                    if (globalIcon) globalIcon.textContent = '🎓';
                }

                setTimeout(checkStatus, 5000);
            } else if (task.status === 'completed') {
                showToast('✅ Training Complete! New model is now active.', 'success');
                if (progressBar) progressBar.style.display = 'none';
                if (progressText) progressText.style.display = 'none';
                if (globalMonitor) globalMonitor.style.display = 'none';

                // Clear state
                state.activeTrainingTaskId = null;
                localStorage.removeItem('activeTrainingTaskId');
            } else if (task.status === 'failed') {
                showToast(`❌ Training Failed: ${task.error}`, 'error');
                if (progressBar) progressBar.style.display = 'none';
                if (progressText) progressText.style.display = 'none';
                if (globalMonitor) globalMonitor.style.display = 'none';

                // Clear state
                state.activeTrainingTaskId = null;
                localStorage.removeItem('activeTrainingTaskId');
            }
        } catch (e) {
            console.error('Task poll error:', e);
            // Don't clear state on temporary network error, let it retry or keep state
        }
    };

    // Ensure UI is visible immediately if resuming
    if (progressBar) progressBar.style.display = 'flex';

    setTimeout(checkStatus, 1000);
}

/* --- Analytics Dashboard & Logging --- */

async function showAnalyticsView() {
    document.getElementById('dashboard-view').style.display = 'none';
    document.getElementById('analytics-view').style.display = 'block';
    document.getElementById('logs-view').style.display = 'none';

    loadAnalyticsStats();
}

async function showLogsView() {
    document.getElementById('dashboard-view').style.display = 'none';
    document.getElementById('analytics-view').style.display = 'none';
    document.getElementById('logs-view').style.display = 'block';

    showLogsTab('activity');
}

function showLogsTab(tab) {
    const activityTab = document.getElementById('logs-activity-tab');
    const errorsTab = document.getElementById('logs-errors-tab');
    const activityBtn = document.getElementById('tab-btn-user-logs');
    const errorsBtn = document.getElementById('tab-btn-error-logs');

    if (tab === 'activity') {
        activityTab.style.display = 'block';
        errorsTab.style.display = 'none';
        activityBtn.className = 'btn btn-primary';
        errorsBtn.className = 'btn btn-secondary';
        loadActivityLogs();
    } else {
        activityTab.style.display = 'none';
        errorsTab.style.display = 'block';
        activityBtn.className = 'btn btn-secondary';
        errorsBtn.className = 'btn btn-primary';
        loadSystemErrorLogs();
    }
}

async function loadAnalyticsStats() {
    try {
        const stats = await apiRequest('/analytics/stats');

        // Update counters
        document.getElementById('stat-projects').textContent = stats.global.projects;
        document.getElementById('stat-images').textContent = stats.global.images;
        document.getElementById('stat-annotations').textContent = stats.global.annotations;
        document.getElementById('stat-users').textContent = stats.global.users;

        // Update distribution bars
        const container = document.getElementById('stats-type-distribution');
        container.innerHTML = '';

        const types = {
            'bbox': { label: 'Bounding Boxes', color: '#3498db' },
            'polygon': { label: 'Polygon Segmentation', color: '#2ecc71' },
            'keypoints': { label: 'Keypoints', color: '#f1c40f' }
        };

        const maxCount = Math.max(...Object.values(stats.by_type), 1);

        Object.keys(types).forEach(type => {
            const count = stats.by_type[type] || 0;
            const percentage = (count / maxCount) * 100;

            const div = document.createElement('div');
            div.style.marginBottom = '15px';
            div.innerHTML = `
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: 600;">${types[type].label}</span>
                    <span style="color: #7f8c8d;">${count} Projects</span>
                </div>
                <div style="height: 12px; background: #ecf0f1; border-radius: 6px; overflow: hidden;">
                    <div style="width: ${percentage}%; height: 100%; background: ${types[type].color}; border-radius: 6px; transition: width 1s ease;"></div>
                </div>
            `;
            container.appendChild(div);
        });

    } catch (error) {
        console.error('Failed to load analytics stats:', error);
    }
}

async function loadActivityLogs() {
    try {
        const logs = await apiRequest('/logs/activity');
        const body = document.getElementById('activity-logs-body');
        body.innerHTML = '';

        if (logs.length === 0) {
            body.innerHTML = '<tr><td colspan="5" style="padding: 30px; text-align: center; color: #95a5a6;">No activity logs found.</td></tr>';
            return;
        }

        logs.forEach(log => {
            const date = new Date(log.created_at);
            const timeStr = date.toLocaleString();

            // Format details
            let detailsStr = '-';
            if (log.details) {
                if (log.action === 'save_annotation') {
                    detailsStr = `Saved ${log.details.count} annotations ${log.details.is_correction ? '(Correction)' : ''}`;
                } else if (log.action === 'create_project' || log.action === 'delete_project') {
                    detailsStr = `Project: ${log.details.name}`;
                } else {
                    detailsStr = JSON.stringify(log.details);
                }
            }

            // Action badge color
            let badgeStyle = 'background: #f1c40f; color: white;'; // default login
            if (log.action.includes('delete')) badgeStyle = 'background: #e74c3c; color: white;';
            if (log.action.includes('create')) badgeStyle = 'background: #2ecc71; color: white;';
            if (log.action.includes('save')) badgeStyle = 'background: #3498db; color: white;';

            const row = document.createElement('tr');
            row.style.borderBottom = '1px solid #edf2f7';
            row.innerHTML = `
                <td style="padding: 12px 20px; font-size: 0.9rem; color: #7f8c8d;">${timeStr}</td>
                <td style="padding: 12px 20px; font-weight: 600; color: #2c3e50;">${log.username}</td>
                <td style="padding: 12px 20px;">
                    <span style="padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: bold; text-transform: uppercase; ${badgeStyle}">
                        ${log.action.replace('_', ' ')}
                    </span>
                </td>
                <td style="padding: 12px 20px; font-size: 0.9rem;">${log.project_name || '-'}</td>
                <td style="padding: 12px 20px; font-size: 0.9rem; color: #34495e;">${detailsStr}</td>
            `;
            body.appendChild(row);
        });

    } catch (error) {
        console.error('Failed to load activity logs:', error);
    }
}

async function loadSystemErrorLogs() {
    try {
        const logs = await apiRequest('/logs/errors');
        const body = document.getElementById('error-logs-body');
        body.innerHTML = '';

        if (logs.length === 0) {
            body.innerHTML = '<tr><td colspan="5" style="padding: 30px; text-align: center; color: #95a5a6;">No system errors logged.</td></tr>';
            return;
        }

        logs.forEach(log => {
            const date = new Date(log.created_at);
            const timeStr = date.toLocaleString();

            const detailsStr = log.details && log.details.error ? log.details.error : JSON.stringify(log.details);

            // Format traceback inside a clean PRE block with a small toggle
            let tracebackHtml = '-';
            if (log.traceback) {
                const tbId = `tb-${log.id}`;
                tracebackHtml = `
                    <button class="btn btn-secondary" style="padding: 2px 6px; font-size: 0.75rem;" onclick="document.getElementById('${tbId}').style.display = document.getElementById('${tbId}').style.display === 'none' ? 'block' : 'none'">View Stack Trace</button>
                    <pre id="${tbId}" style="display: none; background: #2c3e50; color: #ecf0f1; padding: 10px; border-radius: 4px; font-size: 0.75rem; overflow-x: auto; margin-top: 5px;">${log.traceback.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</pre>
                `;
            }

            const row = document.createElement('tr');
            row.style.borderBottom = '1px solid #f8d7da';
            row.innerHTML = `
                <td style="padding: 12px 20px; font-size: 0.9rem; color: #e74c3c; font-weight: 500;">${timeStr}</td>
                <td style="padding: 12px 20px; font-size: 0.9rem; font-weight: bold;">${log.project_name || 'System-wide'}</td>
                <td style="padding: 12px 20px;">
                    <span style="padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: bold; text-transform: uppercase; background: #e74c3c; color: white;">
                        ${log.action.replace('_', ' ')}
                    </span>
                </td>
                <td style="padding: 12px 20px; font-size: 0.9rem; color: #c0392b;">${detailsStr}</td>
                <td style="padding: 12px 20px; width: 300px;">${tracebackHtml}</td>
            `;
            body.appendChild(row);
        });

    } catch (error) {
        console.error('Failed to load error logs:', error);
    }
}

async function showProjectAnalyticsModal(projectId) {
    const modal = document.getElementById('project-analytics-modal');
    const content = document.getElementById('project-analytics-content');

    modal.style.display = 'flex';
    content.innerHTML = '<div style="text-align: center; color: #7f8c8d; padding: 20px;">Loading analytics...</div>';

    try {
        const stats = await apiRequest(`/projects/${projectId}/analytics`);
        document.getElementById('project-analytics-title').textContent = `${stats.project_name} Analytics`;

        content.innerHTML = `
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div class="stat-card" style="background: #fdf2e9; padding: 15px; border-radius: 8px; border-left: 4px solid #e67e22;">
                    <div style="font-size: 0.8rem; color: #d35400; text-transform: uppercase; font-weight: bold;">Human Annotations</div>
                    <div style="font-size: 2rem; font-weight: bold; color: #2c3e50;">${stats.human_annotations}</div>
                    <div style="font-size: 0.75rem; color: #7f8c8d; margin-top: 5px;">Images manually annotated</div>
                </div>
                
                <div class="stat-card" style="background: #e8f8f5; padding: 15px; border-radius: 8px; border-left: 4px solid #1abc9c;">
                    <div style="font-size: 0.8rem; color: #16a085; text-transform: uppercase; font-weight: bold;">AI Annotations</div>
                    <div style="font-size: 2rem; font-weight: bold; color: #2c3e50;">${stats.ai_annotations}</div>
                    <div style="font-size: 0.75rem; color: #7f8c8d; margin-top: 5px;">Images automatically labeled</div>
                </div>
                
                <div class="stat-card" style="background: #eaf2f8; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db;">
                    <div style="font-size: 0.8rem; color: #2980b9; text-transform: uppercase; font-weight: bold;">AI Correct Detections</div>
                    <div style="font-size: 2rem; font-weight: bold; color: #2c3e50;">${stats.ai_correct_detections}</div>
                    <div style="font-size: 0.75rem; color: #7f8c8d; margin-top: 5px;">Images requiring no correction</div>
                </div>
                
                <div class="stat-card" style="background: #fbfcfc; padding: 15px; border-radius: 8px; border-left: 4px solid #95a5a6; border-right: 1px solid #eee; border-top: 1px solid #eee; border-bottom: 1px solid #eee;">
                    <div style="font-size: 0.8rem; color: #7f8c8d; text-transform: uppercase; font-weight: bold;">Estimated Time Saved</div>
                    <div style="font-size: 2rem; font-weight: bold; color: #2c3e50;">${stats.time.time_saved_display}</div>
                    <div style="font-size: 0.75rem; color: #7f8c8d; margin-top: 5px;">vs 100% manual annotation</div>
                </div>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 8px; border: 1px solid #eee; margin-top: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h3 style="margin: 0; font-size: 1rem; color: #2c3e50;">Dataset Verification Status</h3>
                    <div style="font-weight: bold; color: #2ecc71;">${stats.verification.progress_percent}% Verified</div>
                </div>
                
                <div style="height: 12px; background: #ecf0f1; border-radius: 6px; overflow: hidden; display: flex;">
                    <div style="width: ${stats.verification.progress_percent}%; background: #2ecc71; height: 100%;"></div>
                </div>
                
                <div style="display: flex; justify-content: space-between; margin-top: 10px; font-size: 0.85rem; color: #7f8c8d;">
                    <div>✅ Verified: <strong>${stats.verification.verified}</strong></div>
                    <div>✏️ Needs Edit: <strong>${stats.verification.needs_edit}</strong></div>
                    <div>🟡 Unverified: <strong>${stats.verification.unverified}</strong></div>
                </div>
            </div>
        `;
    } catch (e) {
        content.innerHTML = `<div style="text-align: center; color: #e74c3c; padding: 20px;">Failed to load analytics: ${e.message}</div>`;
    }
}
