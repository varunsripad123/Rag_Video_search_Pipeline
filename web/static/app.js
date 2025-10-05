const apiUrlInput = document.getElementById('apiUrl');
const apiKeyInput = document.getElementById('apiKey');
const queryInput = document.getElementById('query');
const sendButton = document.getElementById('send');
const historyDiv = document.getElementById('history');
const resultsDiv = document.getElementById('results');
const statusIndicator = document.getElementById('statusIndicator');
const checkConnectionButton = document.getElementById('checkConnection');

const STORAGE_KEYS = {
  url: 'rag-video-search-api-url',
  key: 'rag-video-search-api-key',
};

const MAX_HISTORY_MESSAGES = 5;
const conversationHistory = [];
const activeObjectUrls = new Set();

function ensureApiBase(value) {
  const trimmed = value.trim();
  if (!trimmed) return '';
  const normalized = trimmed.replace(/\/+$/, '');
  if (normalized.toLowerCase().endsWith('/v1')) {
    return normalized;
  }
  return `${normalized}/v1`;
}

function loadSettings() {
  const savedUrl = localStorage.getItem(STORAGE_KEYS.url);
  const defaultUrl = ensureApiBase(window.location.origin);
  apiUrlInput.value = savedUrl || defaultUrl;
  const savedKey = localStorage.getItem(STORAGE_KEYS.key);
  if (savedKey) {
    apiKeyInput.value = savedKey;
  }
  updateStatus('idle', 'Configure the API endpoint and test the connection.');
}

function persistSettings() {
  const apiBase = ensureApiBase(apiUrlInput.value);
  apiUrlInput.value = apiBase;
  localStorage.setItem(STORAGE_KEYS.url, apiBase);
  localStorage.setItem(STORAGE_KEYS.key, apiKeyInput.value.trim());
}

function getApiBase() {
  return ensureApiBase(apiUrlInput.value);
}

function getApiKey() {
  return apiKeyInput.value.trim();
}

function clearObjectUrls() {
  activeObjectUrls.forEach((url) => URL.revokeObjectURL(url));
  activeObjectUrls.clear();
}

function addMessage(role, content) {
  const message = document.createElement('div');
  message.classList.add('message');
  message.innerHTML = `<strong>${role}:</strong> ${content}`;
  historyDiv.appendChild(message);
  historyDiv.scrollTop = historyDiv.scrollHeight;
}

function trimHistory() {
  const maxEntries = MAX_HISTORY_MESSAGES * 2;
  if (conversationHistory.length > maxEntries) {
    conversationHistory.splice(0, conversationHistory.length - maxEntries);
  }
}

function updateStatus(state, message) {
  statusIndicator.textContent = message;
  statusIndicator.classList.remove('status-ok', 'status-loading');
  if (state === 'ok') {
    statusIndicator.classList.add('status-ok');
  } else if (state === 'loading') {
    statusIndicator.classList.add('status-loading');
  }
}

async function checkConnection() {
  const baseUrl = getApiBase();
  if (!baseUrl) {
    updateStatus('error', 'Enter a valid API endpoint URL.');
    return;
  }
  updateStatus('loading', 'Checking connection…');
  checkConnectionButton.disabled = true;
  try {
    const response = await fetch(`${baseUrl}/health`, {
      headers: {
        'x-api-key': getApiKey() || '',
      },
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || response.statusText);
    }
    const data = await response.json();
    const environment = data.environment ? ` (${data.environment})` : '';
    updateStatus('ok', `Connected${environment}`);
  } catch (error) {
    console.error('Health check failed', error);
    updateStatus('error', `Connection failed: ${error.message}`);
  } finally {
    checkConnectionButton.disabled = false;
  }
}

async function loadPreview(manifestId, videoElement, button) {
  const baseUrl = getApiBase();
  const apiKey = getApiKey();
  if (!baseUrl) {
    addMessage('System', 'Set an API endpoint before loading previews.');
    return;
  }
  if (!apiKey) {
    addMessage('System', 'Provide an API key to request previews.');
    return;
  }
  button.disabled = true;
  const previousLabel = button.textContent;
  button.textContent = 'Loading…';
  try {
    const response = await fetch(`${baseUrl}/decode`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
      },
      body: JSON.stringify({ manifest_id: manifestId, quality: 'preview' }),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || response.statusText);
    }
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    const previousUrl = videoElement.dataset.objectUrl;
    if (previousUrl) {
      URL.revokeObjectURL(previousUrl);
      activeObjectUrls.delete(previousUrl);
    }
    videoElement.dataset.objectUrl = objectUrl;
    activeObjectUrls.add(objectUrl);
    videoElement.src = objectUrl;
    videoElement.load();
    button.textContent = 'Reload preview';
  } catch (error) {
    console.error('Preview failed', error);
    addMessage('System', `Preview error: ${error.message}`);
    button.textContent = previousLabel;
  } finally {
    button.disabled = false;
  }
}

function renderResults(results) {
  clearObjectUrls();
  resultsDiv.innerHTML = '';
  if (!results.length) {
    const empty = document.createElement('p');
    empty.textContent = 'No results returned for this query.';
    resultsDiv.appendChild(empty);
    return;
  }
  results.forEach((item) => {
    const container = document.createElement('div');
    container.classList.add('result-item');

    const heading = document.createElement('h3');
    heading.textContent = `${item.label} — score ${item.score.toFixed(3)}`;
    container.appendChild(heading);

    const meta = document.createElement('div');
    meta.classList.add('meta');
    const truncatedId =
      item.manifest_id.length > 12 ? `${item.manifest_id.slice(0, 8)}…` : item.manifest_id;
    meta.innerHTML = `
      <span>Segment: ${item.start_time.toFixed(2)}s → ${item.end_time.toFixed(2)}s</span>
      <span title="${item.manifest_id}">Manifest ID: ${truncatedId}</span>
    `;
    container.appendChild(meta);

    const previewButton = document.createElement('button');
    previewButton.type = 'button';
    previewButton.textContent = 'Load preview';

    const video = document.createElement('video');
    video.controls = true;
    video.preload = 'none';

    previewButton.addEventListener('click', () => loadPreview(item.manifest_id, video, previewButton));

    container.appendChild(previewButton);
    container.appendChild(video);
    resultsDiv.appendChild(container);
  });
}

async function search(query) {
  const baseUrl = getApiBase();
  const apiKey = getApiKey();
  if (!baseUrl) {
    addMessage('System', 'Set an API endpoint before searching.');
    return;
  }
  if (!apiKey) {
    addMessage('System', 'Provide an API key to perform searches.');
    return;
  }

  const payload = {
    query,
    history: conversationHistory.slice(-MAX_HISTORY_MESSAGES * 2),
    options: {
      expand: true,
      top_k: 5,
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
    addMessage('Assistant', data.answer);
    renderResults(data.results || []);
    conversationHistory.push({ role: 'user', content: query });
    conversationHistory.push({ role: 'assistant', content: data.answer });
    trimHistory();
  } catch (error) {
    console.error('Search failed', error);
    addMessage('System', `Error: ${error.message}`);
  } finally {
    sendButton.disabled = false;
    queryInput.disabled = false;
  }
}

sendButton.addEventListener('click', () => {
  const query = queryInput.value.trim();
  if (!query) return;
  addMessage('User', query);
  queryInput.value = '';
  search(query);
});

queryInput.addEventListener('keypress', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    sendButton.click();
  }
});

checkConnectionButton.addEventListener('click', checkConnection);
apiUrlInput.addEventListener('change', () => {
  persistSettings();
  updateStatus('idle', 'Settings updated — test the connection.');
});
apiKeyInput.addEventListener('change', persistSettings);

window.addEventListener('beforeunload', clearObjectUrls);

loadSettings();

// Automatically test the connection if the user already configured the API URL.
if (apiUrlInput.value) {
  checkConnection();
}
