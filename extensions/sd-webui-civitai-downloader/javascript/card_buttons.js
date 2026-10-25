// card_buttons.js
(function () {
    "use strict";

    const uiText = text => (window.localization && window.localization[text]) || text;

    // Button HTML as a <ul>
    function getButtonsHtml() {
        let html = '<ul class="custom-card-buttons" style="display:none;gap:0.5em;list-style:none;padding:0;margin:0;">';
        if (typeof opts === 'undefined' || opts.civitai_card_button_open_url !== false) {
            html += '<li><button class="open-url-btn" title="Open Civitai page" style="cursor:pointer;">🌐</button></li>';
        }
        if (typeof opts === 'undefined' || opts.civitai_card_button_delete !== false) {
            html += '<li><button class="delete-btn" title="Delete model" style="cursor:pointer;">❌</button></li>';
        }
        html += '</ul>';
        return html;
    }

    function stopEvent(e) {
        e.stopPropagation();
        e.preventDefault();
    }

    function getModelPath(card) {
        // Use cached path if available
        if (card.dataset.civitaiModelPath) return card.dataset.civitaiModelPath;
        const searchTerm = card.querySelector('.actions .additional .search_terms');
        if (searchTerm) return searchTerm.textContent.trim();
        const searchTerm2 = card.querySelector('.actions .additional .search_term');
        if (searchTerm2) return searchTerm2.textContent.trim();
        return null;
    }

    function notifySDWebUI(message, type = "info") {
        message = uiText(message);
        const app = gradioApp();
        const alertBox = app.querySelector('#js_alert_box input, #js_alert_box textarea');
        const alertBtn = app.querySelector('#js_alert_btn button, #js_alert_btn');
        if (alertBox && alertBtn) {
            alertBox.value = JSON.stringify({ message, level: type });
            alertBox.dispatchEvent(new Event("input", { bubbles: true }));
            alertBtn.click();
        } else {
            alert(message);
        }
    }

    function processCard(card) {
        const alreadyHasButtons = !!card.querySelector('.custom-card-buttons');

        if (!alreadyHasButtons) {
            const actions = card.querySelector('.actions');
            const BUTTONS_HTML = getButtonsHtml();
            if (actions) {
                actions.insertAdjacentHTML('beforebegin', BUTTONS_HTML);
            } else {
                card.insertAdjacentHTML('beforeend', BUTTONS_HTML);
            }

            const btns = card.querySelector('.custom-card-buttons');
            const urlBtn = btns.querySelector('.open-url-btn');
            const delBtn = btns.querySelector('.delete-btn');

            if (urlBtn) {
                urlBtn.onclick = async function (e) {
                    e.stopPropagation();
                    e.preventDefault();
                    let modelPath = getModelPath(card);
                    if (!modelPath) {
                        notifySDWebUI("Cannot determine the model path to open its information page.", "error");
                        return;
                    }
                    let metadataPath = modelPath.replace(/\.[^/.]+$/, '.metadata.json').replace(/\\/g, '/');
                    if (!metadataPath.startsWith('models/')) metadataPath = 'models/' + metadataPath;
                    const domain = (typeof opts !== 'undefined' && opts.civitai_preferred_domain)
                        ? opts.civitai_preferred_domain : 'civitai.com';
                    try {
                        let resp = await fetch(`/sd-webui-model-downloader/api/metadata?path=${encodeURIComponent(metadataPath)}`);
                        if (!resp.ok) throw new Error(uiText("Metadata not found"));
                        let metadata = await resp.json();
                        let modelId = null;
                        if (metadata.civitai && metadata.civitai.modelId) {
                            modelId = metadata.civitai.modelId;
                        } else if (metadata.civitai && metadata.civitai.id) {
                            modelId = metadata.civitai.id;
                        } else if (metadata.id) {
                            modelId = metadata.id;
                        }
                        if (!modelId) throw new Error(uiText("No model ID in metadata"));
                        window.open(`https://${domain}/models/${modelId}/`, "_blank");
                    } catch (err) {
                        notifySDWebUI(uiText("Cannot open the model information page: ") + err.message, "error");
                    }
                };
            }
            if (delBtn) {
                delBtn.onclick = async function (e) {
                    stopEvent(e);
                    let rm_confirm = uiText("Are you sure you want to delete this model and its related files? This cannot be undone.");
                    if (!confirm(rm_confirm)) return false;
                    const modelPath = getModelPath(card);
                    if (!modelPath) {
                        notifySDWebUI("Cannot determine the model path to delete.", "error");
                        return false;
                    }
                    let apiModelPath = modelPath.replace(/\\/g, '/');
                    if (!apiModelPath.startsWith('models/')) apiModelPath = 'models/' + apiModelPath;
                    try {
                        const resp = await fetch('/sd-webui-model-downloader/api/delete_model', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ model_path: apiModelPath })
                        });
                        const result = await resp.json();
                        if (resp.ok && result.success) {
                            notifySDWebUI(result.message || 'Model deleted.', 'info');
                            card.remove();
                        } else {
                            notifySDWebUI(result.error || 'Failed to delete model.', 'error');
                        }
                    } catch (err) {
                        notifySDWebUI(uiText('Failed to delete model: ') + err.message, 'error');
                    }
                    return false;
                };
            }
        }

        // Title replacement: run every time so it applies even if opts weren't ready on first pass
        if (typeof opts !== 'undefined' && opts.civitai_show_model_title_on_card && !card.dataset.civitaiTitleSet) {
            card.dataset.civitaiTitleSet = '1';
            (async () => {
                const modelPath = getModelPath(card);
                if (!modelPath) return;
                let metadataPath = modelPath.replace(/\.[^/.]+$/, '.metadata.json').replace(/\\/g, '/');
                if (!metadataPath.startsWith('models/')) metadataPath = 'models/' + metadataPath;
                try {
                    const resp = await fetch(`/sd-webui-model-downloader/api/metadata?path=${encodeURIComponent(metadataPath)}`);
                    if (!resp.ok) return;
                    const metadata = await resp.json();
                    const modelName = metadata && metadata.name ? metadata.name : null;
                    if (!modelName) return;
                    // span.name is the confirmed title element in Forge/A1111
                    const titleEl = card.querySelector('span.name');
                    if (titleEl) titleEl.textContent = modelName;
                } catch (e) { /* ignore */ }
            })();
        }
    }

    function processVisibleCards() {
        const containers = gradioApp().querySelectorAll('.extra-network-cards');
        containers.forEach(container => {
            if (container.offsetParent !== null) {
                const cards = container.querySelectorAll('.card');
                cards.forEach(processCard);
            }
        });
    }

    function setupCardButtonUI() {
        processVisibleCards();
        gradioApp().addEventListener('click', function () {
            setTimeout(processVisibleCards, 100);
        }, true);

        // Watch for DOM changes so buttons appear as soon as cards are rendered
        let observerTimeout = null;
        const observer = new MutationObserver(function (mutationsList) {
            if (observerTimeout) clearTimeout(observerTimeout);
            observerTimeout = setTimeout(() => {
                processVisibleCards();
            }, 150);
        });
        observer.observe(document.body, { childList: true, subtree: true });

        const style = document.createElement('style');
        style.textContent = `
            .card {
                position: relative;
            }
            .card .custom-card-buttons {
                position: absolute;
                right: 34%;
                bottom: 15%;
                z-index: 2;
                padding: 0.2em 0.6em;
                display: none;
                justify-content: center;
                align-items: center;
            }
            .card:hover .custom-card-buttons {
                display: flex !important;
            }
            .custom-card-buttons li {
                display: inline-block;
            }
            .custom-card-buttons button {
                background: var(--button-secondary-background, #eee);
                border: 1px solid var(--button-border-color, #ccc);
                border-radius: 6px;
                padding: 0.2em 0.6em;
                font-size: 1.1em;
                transition: background 0.2s;
                opacity: 0.8;
            }
            .custom-card-buttons button:hover {
                background: var(--button-primary-background, #ccc);
                opacity: 1;
            }
        `;
        document.head.appendChild(style);
    }

    function waitForUiLoadedAndSetup() {
        if (window.onUiLoaded) {
            onUiLoaded(setupCardButtonUI);
        } else {
            setTimeout(waitForUiLoadedAndSetup, 500);
        }
    }

    waitForUiLoadedAndSetup();
})();