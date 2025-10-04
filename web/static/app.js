const apiUrlInput = document.getElementById('apiUrl');
const apiKeyInput = document.getElementById('apiKey');
const queryInput = document.getElementById('query');
const sendButton = document.getElementById('send');
const historyDiv = document.getElementById('history');
const resultsDiv = document.getElementById('results');

const history = [];

function addMessage(role, content) {
  const message = document.createElement('div');
  message.classList.add('message');
  message.innerHTML = `<strong>${role}:</strong> ${content}`;
  historyDiv.appendChild(message);
  historyDiv.scrollTop = historyDiv.scrollHeight;
}

async function search(query) {
  const apiUrl = apiUrlInput.value.replace(/\/$/, '') + '/search';
  const apiKey = apiKeyInput.value;

  const payload = {
    query,
    history,
    options: {
      expand: true,
      top_k: 5,
    },
  };

  try {
    const response = await fetch(apiUrl, {
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
    resultsDiv.innerHTML = '';
    data.results.forEach((item) => {
      const div = document.createElement('div');
      div.classList.add('result-item');
      div.innerHTML = `
        <h3>${item.label} — score ${item.score.toFixed(3)}</h3>
        <p>Segment: ${item.start_time.toFixed(2)}s → ${item.end_time.toFixed(2)}s</p>
        <video src="${item.asset_url}" controls></video>
      `;
      resultsDiv.appendChild(div);
    });

    history.push({ role: 'user', content: query });
    history.push({ role: 'assistant', content: data.answer });
  } catch (error) {
    console.error('Search failed', error);
    addMessage('System', `Error: ${error.message}`);
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
    sendButton.click();
  }
});
