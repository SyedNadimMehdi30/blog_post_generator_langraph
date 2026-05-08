// ========================================
//  Blog Writer Agent — Frontend Logic
// ========================================

let rawMarkdown = '';

// Pipeline step IDs in order
const PIPELINE_STEPS = ['step-router', 'step-research', 'step-orchestrator', 'step-workers', 'step-reducer'];
const CONNECTORS_COUNT = PIPELINE_STEPS.length - 1;
let pipelineInterval = null;

/**
 * Set the topic input from a suggestion chip.
 */
function setTopic(chip) {
    document.getElementById('topic-input').value = chip.textContent.trim();
    document.getElementById('topic-input').focus();
}

/**
 * Simulate the pipeline progress animation.
 */
function startPipelineAnimation() {
    const section = document.getElementById('pipeline-section');
    section.classList.remove('hidden');

    // Reset all steps
    PIPELINE_STEPS.forEach(id => {
        const el = document.getElementById(id);
        el.classList.remove('active', 'done');
    });

    // Reset connectors
    const connectors = document.querySelectorAll('.pipeline-connector');
    connectors.forEach(c => c.classList.remove('done'));

    let current = 0;

    // Activate first step
    document.getElementById(PIPELINE_STEPS[0]).classList.add('active');

    pipelineInterval = setInterval(() => {
        if (current < PIPELINE_STEPS.length) {
            // Mark current as done
            const currentEl = document.getElementById(PIPELINE_STEPS[current]);
            currentEl.classList.remove('active');
            currentEl.classList.add('done');

            // Mark connector as done
            if (current < CONNECTORS_COUNT) {
                connectors[current].classList.add('done');
            }

            current++;

            // Activate next step
            if (current < PIPELINE_STEPS.length) {
                document.getElementById(PIPELINE_STEPS[current]).classList.add('active');
            }
        } else {
            clearInterval(pipelineInterval);
            pipelineInterval = null;
        }
    }, 3500);
}

/**
 * Force-complete all pipeline steps.
 */
function completePipeline() {
    if (pipelineInterval) {
        clearInterval(pipelineInterval);
        pipelineInterval = null;
    }

    PIPELINE_STEPS.forEach(id => {
        const el = document.getElementById(id);
        el.classList.remove('active');
        el.classList.add('done');
    });

    const connectors = document.querySelectorAll('.pipeline-connector');
    connectors.forEach(c => c.classList.add('done'));
}

/**
 * Main generate function — calls the FastAPI backend.
 */
async function generateBlog() {
    const topicInput = document.getElementById('topic-input');
    const btn = document.getElementById('generate-btn');
    const outputSection = document.getElementById('output-section');
    const pipelineSection = document.getElementById('pipeline-section');

    const topic = topicInput.value.trim();
    if (!topic) {
        topicInput.focus();
        topicInput.style.borderColor = '#ef4444';
        setTimeout(() => { topicInput.style.borderColor = ''; }, 1500);
        return;
    }

    // Disable button, show loading state
    btn.disabled = true;
    btn.classList.add('loading');

    // Hide previous output
    outputSection.classList.add('hidden');

    // Start pipeline animation
    startPipelineAnimation();

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic }),
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => null);
            throw new Error(errData?.detail || `Server returned ${response.status}`);
        }

        const data = await response.json();
        rawMarkdown = data.markdown;

        // Complete pipeline
        completePipeline();

        // Render markdown
        renderOutput(data.title, rawMarkdown);
    } catch (err) {
        console.error('Generation failed:', err);
        completePipeline();
        showError(err.message);
    } finally {
        btn.disabled = false;
        btn.classList.remove('loading');
    }
}

/**
 * Render the blog output using marked.js and highlight.js.
 */
function renderOutput(title, markdown) {
    const outputSection = document.getElementById('output-section');
    const outputBody = document.getElementById('output-body');
    const blogTitle = document.getElementById('blog-title');

    // Configure marked
    marked.setOptions({
        highlight: function (code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return hljs.highlightAuto(code).value;
        },
        breaks: false,
        gfm: true,
    });

    blogTitle.textContent = title || 'Generated Blog Post';
    outputBody.innerHTML = marked.parse(markdown);

    // Show output
    outputSection.classList.remove('hidden');

    // Scroll to output
    setTimeout(() => {
        outputSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);
}

/**
 * Show an error message in the output area.
 */
function showError(message) {
    const outputSection = document.getElementById('output-section');
    const outputBody = document.getElementById('output-body');
    const blogTitle = document.getElementById('blog-title');

    blogTitle.textContent = 'Generation Failed';
    outputBody.innerHTML = `
        <div style="text-align: center; padding: 48px 24px;">
            <div style="font-size: 3rem; margin-bottom: 16px;">⚠️</div>
            <h3 style="color: #ef4444; margin-bottom: 12px;">Something went wrong</h3>
            <p style="color: var(--text-secondary); max-width: 500px; margin: 0 auto;">
                ${escapeHtml(message)}
            </p>
            <p style="color: var(--text-muted); margin-top: 16px; font-size: 0.85rem;">
                Please check that the server is running and your API keys are configured.
            </p>
        </div>
    `;

    outputSection.classList.remove('hidden');
}

/**
 * Copy the raw markdown to clipboard.
 */
async function copyMarkdown() {
    if (!rawMarkdown) return;

    const btn = document.getElementById('copy-btn');
    try {
        await navigator.clipboard.writeText(rawMarkdown);
        btn.classList.add('copied');
        btn.querySelector('span').textContent = 'Copied!';
        setTimeout(() => {
            btn.classList.remove('copied');
            btn.querySelector('span').textContent = 'Copy';
        }, 2000);
    } catch {
        // Fallback
        const ta = document.createElement('textarea');
        ta.value = rawMarkdown;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        btn.classList.add('copied');
        btn.querySelector('span').textContent = 'Copied!';
        setTimeout(() => {
            btn.classList.remove('copied');
            btn.querySelector('span').textContent = 'Copy';
        }, 2000);
    }
}

/**
 * Download the raw markdown as a .md file.
 */
function downloadMarkdown() {
    if (!rawMarkdown) return;

    const titleEl = document.getElementById('blog-title');
    const filename = (titleEl.textContent || 'blog-post')
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '_')
        .replace(/(^_|_$)/g, '') + '.md';

    const blob = new Blob([rawMarkdown], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Escape HTML to prevent XSS in error messages.
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Allow Enter key to trigger generation
document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('topic-input');
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            generateBlog();
        }
    });
});
