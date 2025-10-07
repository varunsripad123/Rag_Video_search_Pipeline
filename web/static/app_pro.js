// Professional Memory Search UI

// DOM Elements
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const loadingState = document.getElementById('loadingState');
const resultsSection = document.getElementById('resultsSection');
const resultsGrid = document.getElementById('resultsGrid');
const resultCount = document.getElementById('resultCount');
const searchTime = document.getElementById('searchTime');
const settingsToggle = document.getElementById('settingsToggle');
const settingsContent = document.getElementById('settingsContent');
const apiUrl = document.getElementById('apiUrl');
const apiKey = document.getElementById('apiKey');
const topK = document.getElementById('topK');

// State
let isSearching = false;

// Settings toggle
settingsToggle.addEventListener('click', () => {
    const isHidden = settingsContent.style.display === 'none';
    settingsContent.style.display = isHidden ? 'grid' : 'none';
});

// Suggestion chips
document.querySelectorAll('.suggestion-chip').forEach(chip => {
    chip.addEventListener('click', () => {
        searchInput.value = chip.dataset.query;
        performSearch();
    });
});

// Search on Enter
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        performSearch();
    }
});

// Search button click
searchBtn.addEventListener('click', performSearch);

// Main search function
async function performSearch() {
    const query = searchInput.value.trim();
    
    if (!query || isSearching) return;
    
    isSearching = true;
    
    // Show loading state
    loadingState.style.display = 'block';
    resultsSection.style.display = 'none';
    searchBtn.disabled = true;
    
    const startTime = performance.now();
    
    try {
        const response = await fetch(`${apiUrl.value}/v1/search/similar`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': apiKey.value
            },
            body: JSON.stringify({
                query: query,
                history: [],
                options: {
                    expand: false,
                    top_k: parseInt(topK.value) || 6
                }
            })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        const duration = Math.round(performance.now() - startTime);
        
        displayResults(data.results, duration);
        
    } catch (error) {
        console.error('Search failed:', error);
        showError(error.message);
    } finally {
        isSearching = false;
        searchBtn.disabled = false;
        loadingState.style.display = 'none';
    }
}

// Display search results
function displayResults(results, duration) {
    resultsGrid.innerHTML = '';
    
    if (!results || results.length === 0) {
        resultsGrid.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 60px 20px;">
                <div style="font-size: 48px; margin-bottom: 16px;">🔍</div>
                <h3 style="font-size: 20px; margin-bottom: 8px;">No results found</h3>
                <p style="color: var(--text-secondary);">Try a different search query</p>
            </div>
        `;
        resultsSection.style.display = 'block';
        return;
    }
    
    // Update header
    resultCount.textContent = `${results.length} result${results.length !== 1 ? 's' : ''}`;
    searchTime.textContent = duration;
    
    // Create result cards
    results.forEach((item, index) => {
        const card = createResultCard(item, index);
        resultsGrid.appendChild(card);
    });
    
    resultsSection.style.display = 'block';
}

// Create a result card
function createResultCard(item, index) {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.style.animationDelay = `${index * 0.05}s`;
    
    const score = (item.score * 100).toFixed(1);
    const videoUrl = `${apiUrl.value}/v1/video/${item.manifest_id}?api_key=${apiKey.value}`;
    const videoName = item.label || 'Unknown';
    
    // Build auto-labels if available
    let autoLabelsHtml = '';
    if (item.auto_labels) {
        const labels = [];
        if (item.auto_labels.actions && item.auto_labels.actions.length > 0) {
            labels.push(...item.auto_labels.actions.slice(0, 3));
        }
        if (item.auto_labels.objects && item.auto_labels.objects.length > 0) {
            labels.push(...item.auto_labels.objects.slice(0, 2));
        }
        
        if (labels.length > 0) {
            autoLabelsHtml = `
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px;">
                    ${labels.map(label => `
                        <span style="padding: 4px 10px; background: var(--bg-secondary); 
                                   border-radius: 100px; font-size: 12px; color: var(--text-secondary);">
                            ${label}
                        </span>
                    `).join('')}
                </div>
            `;
        }
    }
    
    card.innerHTML = `
        <div class="result-video">
            <video controls preload="metadata">
                <source src="${videoUrl}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        <div class="result-content">
            <h3 class="result-title">${videoName}</h3>
            <div class="result-meta">
                <span class="result-score">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <path d="M8 2L10 6L14 6.5L11 9.5L11.5 14L8 12L4.5 14L5 9.5L2 6.5L6 6L8 2Z" 
                              fill="currentColor"/>
                    </svg>
                    ${score}%
                </span>
                <span>${item.start_time?.toFixed(1) || 0}s - ${item.end_time?.toFixed(1) || 0}s</span>
            </div>
            ${autoLabelsHtml}
            <div style="margin-top: 12px; display: flex; gap: 8px;">
                <button class="icon-btn" onclick="downloadVideo('${item.manifest_id}', '${videoName}')" 
                        title="Download">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <path d="M8 2V10M8 10L5 7M8 10L11 7M2 12V13C2 13.5523 2.44772 14 3 14H13C13.5523 14 14 13.5523 14 13V12" 
                              stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                    </svg>
                </button>
                <button class="icon-btn" onclick="shareVideo('${item.manifest_id}')" title="Share">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <circle cx="12" cy="4" r="2" stroke="currentColor" stroke-width="1.5"/>
                        <circle cx="4" cy="8" r="2" stroke="currentColor" stroke-width="1.5"/>
                        <circle cx="12" cy="12" r="2" stroke="currentColor" stroke-width="1.5"/>
                        <path d="M6 7L10 5M6 9L10 11" stroke="currentColor" stroke-width="1.5"/>
                    </svg>
                </button>
            </div>
        </div>
    `;
    
    return card;
}

// Download video
async function downloadVideo(manifestId, filename) {
    try {
        const url = `${apiUrl.value}/v1/video/${manifestId}?api_key=${apiKey.value}`;
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `${filename}.mp4`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        showNotification('Download started', 'success');
    } catch (error) {
        console.error('Download failed:', error);
        showNotification('Download failed', 'error');
    }
}

// Share video
function shareVideo(manifestId) {
    const url = `${window.location.origin}/video/${manifestId}`;
    
    if (navigator.share) {
        navigator.share({
            title: 'Memory Search Result',
            url: url
        }).catch(err => console.log('Share failed:', err));
    } else {
        // Fallback: copy to clipboard
        navigator.clipboard.writeText(url).then(() => {
            showNotification('Link copied to clipboard', 'success');
        }).catch(err => {
            console.error('Copy failed:', err);
            showNotification('Failed to copy link', 'error');
        });
    }
}

// Show error message
function showError(message) {
    resultsGrid.innerHTML = `
        <div style="grid-column: 1/-1; text-align: center; padding: 60px 20px;">
            <div style="font-size: 48px; margin-bottom: 16px;">⚠️</div>
            <h3 style="font-size: 20px; margin-bottom: 8px; color: var(--error);">Search Failed</h3>
            <p style="color: var(--text-secondary); margin-bottom: 16px;">${message}</p>
            <button class="btn-secondary" onclick="location.reload()">Retry</button>
        </div>
    `;
    resultsSection.style.display = 'block';
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        bottom: 24px;
        right: 24px;
        padding: 16px 24px;
        background: ${type === 'success' ? 'var(--success)' : type === 'error' ? 'var(--error)' : 'var(--primary)'};
        color: white;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-lg);
        z-index: 1000;
        animation: slideIn 0.3s ease;
        font-weight: 500;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Memory Search initialized');
    
    // Focus search input
    searchInput.focus();
    
    // Check API health
    checkAPIHealth();
});

// Check API health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${apiUrl.value}/v1/health`);
        if (response.ok) {
            console.log('✅ API is healthy');
        }
    } catch (error) {
        console.warn('⚠️ API health check failed:', error);
    }
}
