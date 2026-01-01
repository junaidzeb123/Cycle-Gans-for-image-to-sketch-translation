'use strict';

const API_BASE = 'http://localhost:8000';

const form = document.getElementById('upload-form');
const imageInput = document.getElementById('image-input');
const dropzone = document.getElementById('dropzone');
const dropzoneText = document.getElementById('dropzone-text');
const statusEl = document.getElementById('status');
const previewWrapper = document.getElementById('preview');
const previewImg = document.getElementById('preview-img');
const resultWrapper = document.getElementById('result');
const resultImg = document.getElementById('result-img');
const downloadLink = document.getElementById('download-link');
const resultInfo = document.getElementById('result-info');
const resultPlaceholder = document.getElementById('result-placeholder');
const resetBtn = document.getElementById('reset-btn');
const healthBtn = document.getElementById('health-check');
const healthStatus = document.getElementById('health-status');

let latestObjectUrl = null;

function setStatus(message, type = 'info') {
    statusEl.textContent = message;
    statusEl.dataset.type = type;
}

function setResultInfo(detected, generated) {
    resultInfo.textContent = `Detected domain: ${detected} → Generated: ${generated}`;
}

function clearResult() {
    if (latestObjectUrl) {
        URL.revokeObjectURL(latestObjectUrl);
        latestObjectUrl = null;
    }
    resultWrapper.classList.add('hidden');
    resultPlaceholder.classList.remove('hidden');
    downloadLink.href = '#';
    resultImg.src = '';
}

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = () => {
        previewImg.src = reader.result;
        previewWrapper.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

function resetForm() {
    form.reset();
    previewWrapper.classList.add('hidden');
    previewImg.src = '';
    dropzoneText.textContent = 'Drag & drop an image or click to browse';
    setStatus('No image selected.');
    clearResult();
}

function buildFormData() {
    const file = imageInput.files && imageInput.files[0];
    if (!file) {
        throw new Error('Please choose an image first.');
    }
    const formData = new FormData();
    formData.append('file', file, file.name);
    return formData;
}

async function translate() {
    setStatus('Uploading image and waiting for translation…', 'pending');
    clearResult();

    const formData = buildFormData();
    const endpoint = `${API_BASE}/translate`;

    const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || 'Translation failed.');
    }

    const blob = await response.blob();
    const detected = response.headers.get('X-Detected-Domain') || 'unknown';
    const generated = response.headers.get('X-Generated-Domain') || 'unknown';

    latestObjectUrl = URL.createObjectURL(blob);
    resultImg.src = latestObjectUrl;
    downloadLink.href = latestObjectUrl;
    downloadLink.download = `${generated || 'output'}.png`;

    setResultInfo(detected, generated);
    resultPlaceholder.classList.add('hidden');
    resultWrapper.classList.remove('hidden');
    setStatus('Translation completed successfully!', 'success');
}

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    try {
        await translate();
    } catch (error) {
        console.error(error);
        setStatus(error.message || 'Something went wrong.', 'error');
    }
});

resetBtn.addEventListener('click', () => {
    resetForm();
});

dropzone.addEventListener('click', () => imageInput.click());

dropzone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropzone.classList.add('active');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('active');
});

dropzone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropzone.classList.remove('active');
    if (!event.dataTransfer || !event.dataTransfer.files.length) {
        return;
    }
    const file = event.dataTransfer.files[0];
    imageInput.files = event.dataTransfer.files;
    dropzoneText.textContent = file.name;
    setStatus(`Selected: ${file.name}`);
    showPreview(file);
    clearResult();
});

imageInput.addEventListener('change', () => {
    const file = imageInput.files && imageInput.files[0];
    if (!file) {
        resetForm();
        return;
    }
    dropzoneText.textContent = file.name;
    setStatus(`Selected: ${file.name}`);
    showPreview(file);
    clearResult();
});

healthBtn.addEventListener('click', async () => {
    healthStatus.textContent = 'Checking…';
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) {
            throw new Error('Health check failed');
        }
        const payload = await response.json();
        healthStatus.textContent = `Status: ${payload.status} (device: ${payload.device})`;
    } catch (error) {
        console.error(error);
        healthStatus.textContent = 'Unable to reach API.';
    }
});

resetForm();
