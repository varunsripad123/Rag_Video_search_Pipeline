// DOM Elements
const apiUrlInput = document.getElementById('apiUrl');
const apiKeyInput = document.getElementById('apiKey');
const topKInput = document.getElementById('topK');
const queryInput = document.getElementById('query');
const sendButton = document.getElementById('send');
const resultsDiv = document.getElementById('results');
const loadingState = document.getElementById('loadingState');
const resultsSection = document.getElementById('resultsSection');
const resultCount = document.getElementById('resultCount');
const searchDuration = document.getElementById('searchDuration');
const settingsToggle = document.getElementById('settingsToggle');
const settingsContent = document.getElementById('settingsContent');

// Stats elements
const totalVideosEl = document.getElementById('totalVideos');
const indexSizeEl = document.getElementById('indexSize');
const searchTimeEl = document.getElementById('searchTime');

// State
let searchTimes = [];
let isSearching = false;

// Settings toggle
settingsToggle.addEventListener('click', () => {
  const isHidden = settingsContent.style.display === 'none' || !settingsContent.style.display;
  settingsContent.style.display = isHidden ? 'block' : 'none';
});

// Initialize - hide settings by default
settingsContent.style.display = 'none';

// Suggestion chips
document.querySelectorAll('.suggestion-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    queryInput.value = chip.dataset.query;
    queryInput.focus();
  });
});

// Load stats on page load
async function loadStats() {
  try {
    const apiUrl = apiUrlInput.value.replace(/\/$/, '');
    const response = await fetch(`${apiUrl}/v1/health`, {
      headers: {
        'x-api-key': apiKeyInput.value,
      },
    });
    
    if (response.ok) {
      const data = await response.json();
      totalVideosEl.textContent = data.index_size || '272';
      indexSizeEl.textContent = formatBytes(data.index_size * 1288 * 4); // Approximate
      searchTimeEl.textContent = searchTimes.length > 0 
        ? `${Math.round(searchTimes.reduce((a, b) => a + b, 0) / searchTimes.length)}ms`
        : '--';
    }
  } catch (error) {
    console.log('Could not load stats:', error);
    totalVideosEl.textContent = '272';
    indexSizeEl.textContent = '~1.4MB';
  }
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Search function
async function search(query) {
  if (isSearching) return;
  
  isSearching = true;
  loadingState.style.display = 'flex';
  resultsSection.style.display = 'none';
  sendButton.disabled = true;

  const apiUrl = apiUrlInput.value.replace(/\/$/, '') + '/v1/search/similar';
  const apiKey = apiKeyInput.value;
  const topK = parseInt(topKInput.value) || 5;

  const startTime = performance.now();

  const payload = {
    query,
    history: [],
    options: {
      expand: false,
      top_k: topK,
    },
  };

  sendButton.disabled = true;
  queryInput.disabled = true;
  addMessage('System', 'Searching…');

  try {
    const response = await fetch(`${baseUrl}/search/similar`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || response.statusText);
    }

    const data = await response.json();
    const endTime = performance.now();
    const duration = Math.round(endTime - startTime);
    
    searchTimes.push(duration);
    if (searchTimes.length > 10) searchTimes.shift();

    displayResults(data.results, duration);
    updateStats();
  } catch (error) {
    console.error('Search failed', error);
    showError(error.message);
  } finally {
    isSearching = false;
    loadingState.style.display = 'none';
    sendButton.disabled = false;
  }
}

function displayResults(results, duration) {
  resultsDiv.innerHTML = '';
  
  if (!results || results.length === 0) {
    resultsDiv.innerHTML = `
      <div class="no-results">
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <circle cx="11" cy="11" r="8"/>
          <path d="m21 21-4.35-4.35"/>
        </svg>
        <h3>No results found</h3>
        <p>Try a different search query</p>
      </div>
    `;
    resultsSection.style.display = 'block';
    return;
  }

  resultCount.textContent = `${results.length} result${results.length !== 1 ? 's' : ''}`;
  searchDuration.textContent = `${duration}ms`;

  results.forEach((item, index) => {
    const div = document.createElement('div');
    div.classList.add('result-card');
    div.style.animationDelay = `${index * 0.05}s`;
    
    const score = (item.score * 100).toFixed(1);
    const videoPath = item.asset_url || '';
    const videoName = videoPath.split('/').pop().split('\\').pop() || item.label || 'Unknown';
    
    // Build auto-labels display if available
    let autoLabelsHtml = '';
    if (item.auto_labels) {
      const objects = item.auto_labels.objects || [];
      const action = item.auto_labels.action || '';
      const caption = item.auto_labels.caption || '';
      
      if (objects.length > 0 || action || caption) {
        autoLabelsHtml = `<div class="auto-labels">`;
        if (objects.length > 0) {
          autoLabelsHtml += `<div class="label-group"><strong>Objects:</strong> ${objects.slice(0, 3).join(', ')}</div>`;
        }
        if (action && action !== 'static scene' && action !== 'unknown') {
          autoLabelsHtml += `<div class="label-group"><strong>Action:</strong> ${action}</div>`;
        }
        if (caption && caption.length > 10) {
          autoLabelsHtml += `<div class="label-group"><strong>Caption:</strong> ${caption}</div>`;
        }
        autoLabelsHtml += `</div>`;
      }
    }
    
    div.innerHTML = `
      <div class="result-header">
        <div class="result-info">
          <h3 class="result-title">${videoName}</h3>
          <div class="result-meta">
            <span class="meta-item">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <circle cx="12" cy="12" r="10"/>
                <polyline points="12 6 12 12 16 14"/>
              </svg>
              ${item.start_time?.toFixed(1) || 0}s - ${item.end_time?.toFixed(1) || 2}s
            </span>
            <span class="meta-item">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
              </svg>
              ${item.label || 'Unknown'}
            </span>
            <button class="download-btn" onclick="downloadVideo('${item.manifest_id}', '${videoName}')" title="Download Original Video">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
              </svg>
            </button>
          </div>
        </div>
        <div class="result-score">
          <div class="score-circle">
            <span class="score-value">${score}%</span>
          </div>
        </div>
      </div>
      <div class="result-video">
        <video controls preload="metadata" width="100%">
          <source src="${apiUrlInput.value.replace(/\/$/, '')}/v1/video/${item.manifest_id}?api_key=${apiKeyInput.value}" type="video/mp4">
          Your browser does not support the video tag.
        </video>
      </div>
      ${autoLabelsHtml}
      <div class="result-footer">
        <span class="label-tag">${item.label || 'Unknown'}</span>
      </div>
    `;
    
    resultsDiv.appendChild(div);
  });

  resultsSection.style.display = 'block';
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(message) {
  resultsDiv.innerHTML = `
    <div class="error-state">
      <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <circle cx="12" cy="12" r="10"/>
        <line x1="15" y1="9" x2="9" y2="15"/>
        <line x1="9" y1="9" x2="15" y2="15"/>
      </svg>
      <h3>Search Failed</h3>
      <p>${message}</p>
      <button onclick="location.reload()" class="retry-button">Retry</button>
    </div>
  `;
  resultsSection.style.display = 'block';
}

function updateStats() {
  if (searchTimes.length > 0) {
    const avgTime = Math.round(searchTimes.reduce((a, b) => a + b, 0) / searchTimes.length);
    searchTimeEl.textContent = `${avgTime}ms`;
  }
}

// Event listeners
sendButton.addEventListener('click', () => {
  const query = queryInput.value.trim();
  if (!query) return;
  search(query);
});

queryInput.addEventListener('keypress', (event) => {
  if (event.key === 'Enter' && !isSearching) {
    const query = queryInput.value.trim();
    if (query) search(query);
  }
});

// Download video function
async function downloadVideo(manifestId, filename) {
  try {
    const apiUrl = apiUrlInput.value.replace(/\/$/, '');
    const apiKey = apiKeyInput.value;
    
    const url = `${apiUrl}/v1/video/${manifestId}?api_key=${apiKey}`;
    
    // Create a temporary link to trigger download
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.target = '_blank';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    console.log(`Downloading: ${filename}`);
  } catch (error) {
    console.error('Download failed:', error);
    alert('Failed to download video: ' + error.message);
  }
}

// Load stats on page load
loadStats();
