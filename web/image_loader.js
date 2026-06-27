import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "VantageMultiImageLoader",
    async nodeCreated(node) {
        if (node.comfyClass !== "VantageMultiImageLoader") return;

        const GRID_GAP = 8;
        const MIN_GALLERY_HEIGHT = 250;
        const V3_MIN_WIDTH = 100;
        const V1_MIN_WIDTH = 220;
        const V3_PADDING_BOTTOM = 15;
        const V1_PADDING_BOTTOM = 25;
        const SWAP_THROTTLE_MS = 50;
        const SWAP_MOVE_THRESHOLD = 5;

        let v3NodeElement = null;
        let v3EventsAttached = false;
        let isLayouting = false;
        let draggedNode = null;
        let lastSwapX = 0;
        let lastSwapY = 0;
        let lastSwapTime = 0;
        let lastObservedWidth = 0;
        let lastObservedHeight = 0;

        const container = document.createElement("div");
        container.style.cssText = `
            width: 100%;
            min-height: ${MIN_GALLERY_HEIGHT}px;
            min-width: 100px;
            background: #222222;
            border: 1px solid #353545;
            border-radius: 4px;
            margin-top: 5px;
            padding: 10px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            gap: 10px;
            pointer-events: auto;
            overflow: hidden;
        `;

        const topBar = document.createElement("div");
        topBar.style.cssText = "display:flex; flex-wrap:wrap; justify-content:flex-start; align-items:center; width:100%; gap:8px;";

        const uploadBtn = document.createElement("button");
        uploadBtn.innerText = "Upload Images";
        uploadBtn.title = "Upload one or more images into this node gallery.";
        uploadBtn.style.cssText = `
            background:#3a3f4b; color:white; border:1px solid #5a5f6b;
            padding:3px 8px; border-radius:3px; cursor:pointer; font-size:10px;
        `;

        const removeAllBtn = document.createElement("button");
        removeAllBtn.innerText = "Remove All";
        removeAllBtn.title = "Clear every image from the gallery and reset the hidden path list.";
        removeAllBtn.style.cssText = `
            background:#cc2222; color:white; border:1px solid #aa1111;
            padding:3px 8px; border-radius:3px; cursor:pointer; font-size:10px;
            transition:background 0.2s;
        `;
        removeAllBtn.onmouseenter = () => { removeAllBtn.style.background = "#ff3333"; };
        removeAllBtn.onmouseleave = () => { removeAllBtn.style.background = "#cc2222"; };

        const gridWrapper = document.createElement("div");
        gridWrapper.style.cssText = `
            position: relative;
            flex-grow: 1;
            width: 100%;
            min-height: 0;
        `;

        const grid = document.createElement("div");
        grid.style.cssText = `
            position: absolute;
            inset: 0;
            display: grid;
            gap: ${GRID_GAP}px;
            justify-content: center;
            align-content: center;
        `;

        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.multiple = true;
        fileInput.accept = "image/*";
        fileInput.style.display = "none";

        topBar.appendChild(uploadBtn);
        topBar.appendChild(removeAllBtn);
        gridWrapper.appendChild(grid);
        container.appendChild(topBar);
        container.appendChild(gridWrapper);
        container.appendChild(fileInput);

        const galleryWidget = node.addDOMWidget("Gallery", "html_gallery", container, { serialize: false });
        const pathsWidget = node.widgets?.find((w) => w.name === "image_paths");
        const oldCallback = pathsWidget?.callback;

        function getPaths() {
            return (pathsWidget?.value || "")
                .split(/\n|,/) 
                .map((s) => s.trim())
                .filter(Boolean);
        }

        function getNodeMetrics(widthHint = 220) {
            const isV3 = checkIsV3();
            const minW = isV3 ? V3_MIN_WIDTH : V1_MIN_WIDTH;
            const paddingBottom = isV3 ? V3_PADDING_BOTTOM : V1_PADDING_BOTTOM;
            const galleryY = galleryWidget.last_y || 40;
            const minOutputsHeight = (node.outputs ? node.outputs.length : 1) * 20;
            const absoluteMinHeight = Math.max(galleryY + MIN_GALLERY_HEIGHT + paddingBottom, minOutputsHeight + 40);
            const nodeWidth = node.size?.[0] || widthHint || 220;
            return { isV3, minW, paddingBottom, galleryY, absoluteMinHeight, nodeWidth };
        }

        function checkIsV3() {
            if (v3NodeElement) return true;
            let el = container.parentElement;
            while (el) {
                if (
                    (el.tagName && el.tagName.toLowerCase().includes("comfy-node")) ||
                    (el.classList && el.classList.contains("comfy-node"))
                ) {
                    v3NodeElement = el;
                    return true;
                }
                el = el.parentElement || (el.getRootNode ? el.getRootNode().host : null);
            }
            return false;
        }

        function hidePathsWidget() {
            if (!pathsWidget) return;

            Object.defineProperty(pathsWidget, "hidden", {
                get: () => true,
                set: () => {},
            });
            Object.defineProperty(pathsWidget, "type", {
                get: () => "hidden",
                set: () => {},
            });

            pathsWidget.computeSize = () => [0, 0];

            const hideInterval = setInterval(() => {
                if (pathsWidget.element) {
                    pathsWidget.element.style.display = "none";
                }
            }, 50);
            setTimeout(() => clearInterval(hideInterval), 1000);
        }

        function setWidgetValue(newPathsArray, isRearranging = false) {
            if (!pathsWidget) return;
            const val = newPathsArray.join("\n");

            const tempCallback = pathsWidget.callback;
            pathsWidget.callback = null;
            pathsWidget.value = val;
            if (oldCallback) oldCallback.apply(pathsWidget, [val]);
            pathsWidget.callback = tempCallback;

            refreshGallery(isRearranging);
        }

        function notifyConnectedNodes(imageCount) {
            if (!node.outputs) return;
            for (const output of node.outputs) {
                if (!output.links) continue;
                for (const linkId of output.links) {
                    const link = app.graph.links[linkId];
                    if (!link) continue;
                    const targetNode = app.graph.getNodeById(link.target_id);
                    if (targetNode && typeof targetNode._syncImageCount === "function") {
                        targetNode._syncImageCount(imageCount);
                    }
                }
            }
        }

        // Kept as a stub so existing downstream code still has a stable function entry point
        // even though the dynamic output mutation is currently disabled.
        function syncOutputs() {}

        function optimizeGrid(gridW, gridH) {
            const paths = getPaths();
            const count = paths.length;

            if (count === 0) {
                grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(75px, 1fr))";
                grid.style.gridAutoRows = "max-content";
                return;
            }

            if (gridW <= 0 || gridH <= 0) return;

            let bestSize = 0;
            let bestCols = 1;

            for (let cols = 1; cols <= count; cols++) {
                const rows = Math.ceil(count / cols);
                const maxW = Math.max(5, (gridW - (cols - 1) * GRID_GAP) / cols);
                const maxH = Math.max(5, (gridH - (rows - 1) * GRID_GAP) / rows);
                const size = Math.min(maxW, maxH);

                // Favor more columns when multiple layouts yield nearly the same tile size.
                if (size >= bestSize - 0.1) {
                    bestSize = size;
                    bestCols = cols;
                }
            }

            bestSize = Math.max(10, Math.floor(bestSize));
            grid.style.gridTemplateColumns = `repeat(${bestCols}, ${bestSize}px)`;
            grid.style.gridAutoRows = `${bestSize}px`;
        }

        function enforceV3CSS() {
            if (!checkIsV3() || !v3NodeElement) return;
            const { absoluteMinHeight } = getNodeMetrics();
            v3NodeElement.style.removeProperty("min-width");
            v3NodeElement.style.setProperty("min-height", `${absoluteMinHeight}px`, "important");

            if (!v3EventsAttached) {
                v3EventsAttached = true;
                v3NodeElement.addEventListener("dragover", (e) => e.preventDefault());
                v3NodeElement.addEventListener("drop", (e) => {
                    if (!e.dataTransfer?.files) return;
                    const files = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith("image/"));
                    if (!files.length) return;
                    e.preventDefault();
                    e.stopPropagation();
                    handleFiles(files);
                });
            }
        }

        function syncLayoutToNode() {
            const nodeWidth = node.size?.[0] || 220;
            const targetWidth = Math.max(10, nodeWidth - 30);
            container.style.width = `${targetWidth}px`;
            container.style.maxWidth = `${targetWidth}px`;
            container.style.boxSizing = "border-box";
        }

        function updateLayout(forceShrink = false) {
            if (isLayouting) return;
            isLayouting = true;

            const { minW, paddingBottom, galleryY, absoluteMinHeight } = getNodeMetrics();
            node.min_size = [minW, absoluteMinHeight];
            enforceV3CSS();

            let targetW = Math.max(node.size?.[0] || minW, minW);
            let targetH = forceShrink ? absoluteMinHeight : (node.size?.[1] || absoluteMinHeight);
            targetH = Math.max(targetH, absoluteMinHeight);

            if ((node.size?.[0] || 0) !== targetW || (node.size?.[1] || 0) !== targetH) {
                node.setSize([targetW, targetH]);
                app.graph.setDirtyCanvas(true, true);
            }

            const availableGalleryHeight = targetH - galleryY - paddingBottom;
            container.style.height = `${availableGalleryHeight}px`;
            isLayouting = false;
        }
        
        function encodePathEntry(path, rmbgEnabled = false) {
            return `${rmbgEnabled ? 1 : 0}|${path}`;
        }

        function decodePathEntry(entry) {
            const pipeIndex = entry.indexOf("|");

            if (pipeIndex === -1) {
                return { isrmbg: false, imgpath: entry };
            }
            return {
                isrmbg: entry.slice(0, pipeIndex) === "1",
                imgpath: entry.slice(pipeIndex + 1),
            };
        }
        
        function getEntries() {
            return (pathsWidget?.value || "")
                .split(/\n|,/)
                .map((s) => s.trim())
                .filter(Boolean);
        }

        function createThumbItem(path, index, paths) {
            const item = document.createElement("div");
            item.dataset.path = path;
            item.draggable = true;
            item.style.cssText = `
                position: relative;
                width: 100%;
                height: 100%;
                aspect-ratio: 1 / 1;
                background: #000000;
                border-radius: 4px;
                border: 1px solid #444;
                overflow: hidden;
                cursor: grab;
                display: flex;
                align-items: center;
                justify-content: center;
            `;
            const { isrmbg, imgpath } = decodePathEntry(path);
            const img = document.createElement("img");
            img.src = `/api/view?filename=${encodeURIComponent(imgpath)}&type=input`;
            img.draggable = false;
            img.style.cssText = "max-width:100%; max-height:100%; object-fit:contain; pointer-events:auto; display:block;";

            const del = document.createElement("div");
            del.innerHTML = "×";
            del.title = "Remove this image from the gallery.";
            del.style.cssText = `
                position: absolute; top: 0; right: 0;
                background: #cc2222; color: white;
                width: 18px; height: 18px;
                display: flex; align-items: center; justify-content: center;
                font-size: 14px; cursor: pointer; z-index: 10;
                font-family: Arial, sans-serif; font-weight: bold;
                line-height: 1; border-bottom-left-radius: 4px;
                transition: background 0.2s;
            `;
            del.onmouseenter = () => { del.style.background = "#ff3333"; };
            del.onmouseleave = () => { del.style.background = "#cc2222"; };
            del.onclick = (e) => {
                e.stopPropagation();
                setWidgetValue(paths.filter((_, i) => i !== index), false);
            };
            
            const rmbgWrap = document.createElement("div");
            rmbgWrap.title = "Remove background for this image";
            rmbgWrap.style.cssText = `
                position: absolute;
                top: 4px;
                left: 4px;
                width: 20px;
                height: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: rgba(0, 0, 0, 0.75);
                border-radius: 4px;
                z-index: 20;
                pointer-events: auto;
            `;

            const rmbg = document.createElement("input");
            rmbg.type = "checkbox";
            rmbg.checked = isrmbg;
            rmbg.style.cssText = `
                margin: 0;
                width: 14px;
                height: 14px;
                cursor: pointer;
                pointer-events: auto;
            `;

            rmbg.addEventListener("mousedown", (e) => e.stopPropagation());
            rmbg.addEventListener("click", (e) => e.stopPropagation());
            rmbg.addEventListener("dragstart", (e) => {
                e.preventDefault();
                e.stopPropagation();
            });

            rmbg.addEventListener("change", (e) => {
                e.stopPropagation();
                const updatedEntries = getEntries().map((entry, i) => {
                    const decoded = decodePathEntry(entry);
                    if (i === index) {
                        return encodePathEntry(decoded.imgpath, rmbg.checked);
                    }
                    return entry;
                });

                setWidgetValue(updatedEntries, false);
            });

            rmbgWrap.appendChild(rmbg);
            
            const numBadge = document.createElement("div");
            numBadge.innerText = String(index + 1);
            numBadge.style.cssText = `
                position: absolute; bottom: 0; left: 0;
                background: rgba(0, 0, 0, 0.75); color: #fff;
                padding: 2px 6px; font-size: 11px; font-family: sans-serif;
                font-weight: bold; border-top-right-radius: 4px; pointer-events: none;
                z-index: 5;
            `;

            item.addEventListener("contextmenu", (e) => e.stopPropagation());

            item.ondragstart = (e) => {
                draggedNode = item;
                e.dataTransfer.setData("text/plain", path);
                e.dataTransfer.effectAllowed = "move";

                // Delay styling until drag has officially started, otherwise some browsers
                // snapshot the placeholder styling instead of the original thumbnail.
                setTimeout(() => {
                    if (draggedNode !== item) return;
                    item.style.background = "transparent";
                    item.style.border = "2px dashed #666";
                    Array.from(item.children).forEach((c) => { c.style.opacity = "0"; });
                }, 0);
            };

            item.ondragend = () => {
                if (!draggedNode) return;
                draggedNode.style.background = "#000000";
                draggedNode.style.border = "1px solid #444";
                Array.from(draggedNode.children).forEach((c) => { c.style.opacity = "1"; });
                draggedNode = null;

                const newPaths = Array.from(grid.children).map((n) => n.dataset.path);
                const currentVal = (pathsWidget?.value || "").trim();
                if (newPaths.join("\n") !== currentVal) {
                    setWidgetValue(newPaths, true);
                }
            };

            item.ondragover = (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (!draggedNode || draggedNode === item) return;

                const distMoved = Math.hypot(e.clientX - lastSwapX, e.clientY - lastSwapY);
                if (Date.now() - lastSwapTime < SWAP_THROTTLE_MS && distMoved < SWAP_MOVE_THRESHOLD) return;

                const itemRect = item.getBoundingClientRect();
                const bufferX = itemRect.width * 0.25;
                const bufferY = itemRect.height * 0.25;
                if (
                    e.clientX < itemRect.left + bufferX ||
                    e.clientX > itemRect.right - bufferX ||
                    e.clientY < itemRect.top + bufferY ||
                    e.clientY > itemRect.bottom - bufferY
                ) {
                    return;
                }

                const items = Array.from(grid.children);
                const draggedIdx = items.indexOf(draggedNode);
                const targetIdx = items.indexOf(item);
                if (draggedIdx < targetIdx) {
                    grid.insertBefore(draggedNode, item.nextSibling);
                } else {
                    grid.insertBefore(draggedNode, item);
                }

                lastSwapX = e.clientX;
                lastSwapY = e.clientY;
                lastSwapTime = Date.now();
            };

            item.ondrop = (e) => {
                e.preventDefault();
                e.stopPropagation();
            };

            item.appendChild(img);
            item.appendChild(del);
            item.appendChild(numBadge);
            item.appendChild(rmbgWrap);
            return item;
        }

        function refreshGallery(isRearranging = false) {
            grid.innerHTML = "";
            const paths = getPaths();

            if (!isRearranging) syncOutputs(paths.length);

            node._imageCount = paths.length;
            notifyConnectedNodes(paths.length);

            paths.forEach((path, index) => {
                grid.appendChild(createThumbItem(path, index, paths));
            });

            if (!isRearranging) {
                requestAnimationFrame(() => {
                    updateLayout();
                    if (gridWrapper.offsetWidth > 0) {
                        optimizeGrid(gridWrapper.offsetWidth, gridWrapper.offsetHeight);
                    }
                });
            }
        }

        async function handleFiles(files) {
            const uploaded = [];
            for (const file of files) {
                const body = new FormData();
                body.append("image", file);
                try {
                    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
                    if (resp.status === 200) {
                        const data = await resp.json();
                        let name = data.name;
                        if (data.subfolder) name = `${data.subfolder}/${name}`;
                        uploaded.push(name);
                    }
                } catch (e) {
                    console.error("Upload error", e);
                }
            }

            if (uploaded.length > 0) {
                const current = (pathsWidget?.value || "").trim();
                const allPaths = current ? current.split("\n").concat(uploaded) : uploaded;
                setWidgetValue(allPaths, false);
            }
        }

        function installNodeHooks() {
            node.syncLayoutToNode = syncLayoutToNode;

            galleryWidget.computeSize = function(width) {
                const { galleryY, absoluteMinHeight, nodeWidth } = getNodeMetrics(width);
                const minOutputsHeight = (node.outputs ? node.outputs.length : 1) * 20;
                const requiredGalleryHeight = Math.max(MIN_GALLERY_HEIGHT, minOutputsHeight + 40 - galleryY);
                return [Math.max(10, nodeWidth - 30), Math.max(requiredGalleryHeight, absoluteMinHeight - galleryY)];
            };

            const origOnResize = node.onResize;
            node.onResize = function(size) {
                const { minW, paddingBottom, galleryY, absoluteMinHeight } = getNodeMetrics();
                size[0] = Math.max(size[0], minW);
                size[1] = Math.max(size[1], absoluteMinHeight);
                if (origOnResize) origOnResize.call(this, size);
                if (this.syncLayoutToNode) this.syncLayoutToNode();
                if (isLayouting) return;
                this.min_size = [minW, absoluteMinHeight];
                enforceV3CSS();
                container.style.height = `${size[1] - galleryY - paddingBottom}px`;
            };

            const origOnConfigure = node.onConfigure;
            node.onConfigure = function(info) {
                const out = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
                setTimeout(() => {
                    if (this.syncLayoutToNode) this.syncLayoutToNode();
                }, 0);
                return out;
            };

            const origComputeSize = node.computeSize;
            node.computeSize = function(out) {
                const { minW, absoluteMinHeight } = getNodeMetrics();
                const res = origComputeSize ? origComputeSize.apply(this, arguments) : [minW, MIN_GALLERY_HEIGHT];
                this.min_size = [minW, absoluteMinHeight];
                res[0] = Math.max(res[0], minW);
                res[1] = Math.max(res[1], absoluteMinHeight);
                enforceV3CSS();
                return res;
            };

            const origSetSize = node.setSize;
            node.setSize = function(size) {
                const { minW, absoluteMinHeight } = getNodeMetrics();
                size[0] = Math.max(size[0], minW);
                size[1] = Math.max(size[1], absoluteMinHeight);
                if (origSetSize) {
                    origSetSize.call(this, size);
                } else {
                    this.size = size;
                }
                enforceV3CSS();
            };

            const origOnDragDrop = node.onDragDrop;
            node.onDragDrop = function(e) {
                let handled = false;
                if (e.dataTransfer?.files) {
                    const files = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith("image/"));
                    if (files.length > 0) {
                        e.preventDefault();
                        handleFiles(files);
                        handled = true;
                    }
                }
                if (!handled && origOnDragDrop) {
                    return origOnDragDrop.apply(this, arguments);
                }
                return handled;
            };

            const origOnDragOver = node.onDragOver;
            node.onDragOver = function(e) {
                if (e.dataTransfer?.items) {
                    const hasImage = Array.from(e.dataTransfer.items).some((f) => f.kind === "file" && f.type.startsWith("image/"));
                    if (hasImage) {
                        e.preventDefault();
                        return true;
                    }
                }
                if (origOnDragOver) return origOnDragOver.apply(this, arguments);
                return false;
            };

            const origOnRemoved = node.onRemoved;
            node.onRemoved = function() {
                document.removeEventListener("paste", pasteHandler, true);
                resizeObserver.disconnect();
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
            };

            const origOnAdded = node.onAdded;
            node.onAdded = function() {
                if (origOnAdded) origOnAdded.apply(this, arguments);
                if (!checkIsV3()) {
                    requestAnimationFrame(() => {
                        const { absoluteMinHeight } = getNodeMetrics();
                        if (this.size && this.size[1] > absoluteMinHeight + 5) {
                            this.setSize([this.size[0], absoluteMinHeight]);
                            if (app.graph) app.graph.setDirtyCanvas(true, true);
                        }
                    });
                }
            };
        }

        const resizeObserver = new ResizeObserver((entries) => {
            enforceV3CSS();
            for (const entry of entries) {
                const w = Math.round(entry.contentRect.width);
                const h = Math.round(entry.contentRect.height);
                if (Math.abs(w - lastObservedWidth) > 1 || Math.abs(h - lastObservedHeight) > 1) {
                    lastObservedWidth = w;
                    lastObservedHeight = h;
                    if (h > 0) optimizeGrid(w, h);
                }
            }
        });

        function pasteHandler(e) {
            if (!(app.canvas.selected_nodes && app.canvas.selected_nodes[node.id])) return;
            const items = e.clipboardData?.items;
            if (!items) return;

            const files = [];
            for (let i = 0; i < items.length; i++) {
                if (items[i].kind === "file" && items[i].type.startsWith("image/")) {
                    files.push(items[i].getAsFile());
                }
            }

            if (files.length > 0) {
                e.preventDefault();
                e.stopImmediatePropagation();
                handleFiles(files);
            }
        }

        uploadBtn.onclick = () => fileInput.click();
        fileInput.onchange = (e) => handleFiles(e.target.files);
        removeAllBtn.onclick = () => setWidgetValue([], false);

        container.ondragover = (e) => {
            e.preventDefault();
            e.stopPropagation();
            container.style.borderColor = "#4CAF50";
        };
        container.ondragleave = (e) => {
            e.preventDefault();
            e.stopPropagation();
            container.style.borderColor = "#353545";
        };
        container.ondrop = (e) => {
            e.preventDefault();
            e.stopPropagation();
            container.style.borderColor = "#353545";
            if (e.dataTransfer.files.length > 0) handleFiles(e.dataTransfer.files);
        };

        hidePathsWidget();
        installNodeHooks();
        resizeObserver.observe(gridWrapper);
        document.addEventListener("paste", pasteHandler, true);

        if (pathsWidget) {
            pathsWidget.callback = (v) => {
                if (oldCallback) oldCallback.apply(pathsWidget, [v]);
                refreshGallery();
            };
        }

        refreshGallery();
        setTimeout(() => {
            refreshGallery();
            node.syncLayoutToNode();
        }, 100);
    },
});
