import { app } from "../../scripts/app.js";

function hideWidget(node, name) {
    const w = node.widgets?.find(w => w.name === name);
    if (w) {
        w.hidden = true;
        w.computeSize = () => [0, 0];
    }
}

function buildQwenSpeakerBankUI(node) {
    if (!node.properties) node.properties = {};

    const mainSpeakerWidget = node.widgets.find(w => w.name === "speaker_1");
    const jsonWidget = node.widgets.find(w => w.name === "speakers_json");

    if (!mainSpeakerWidget || !jsonWidget) {
        console.warn("[SpeakerBank] Required widgets missing");
        return;
    }

    const voiceOptions = mainSpeakerWidget.options?.values || [];

    /* -------------------------------------------------
     * ONE-TIME INIT (called from node.onConfigure)
     * ------------------------------------------------- */
    if (!node.properties._speakerBankInit) {
        let parsed = { speakers: [] };
        try {
            parsed = JSON.parse(jsonWidget.value);
        } catch {}

        const voices = Array.isArray(parsed.speakers)
            ? parsed.speakers.map(s => s.voice)
            : [];

        node.properties.mainSpeaker =
            voices[0] ?? mainSpeakerWidget.value ?? voiceOptions[0] ?? "";

        node.properties.extraSpeakers = voices.slice(1);
        node.properties._speakerBankInit = true;
    }

    /* -------------------------------------------------
     * KEEP MAIN SPEAKER IN SYNC
     * ------------------------------------------------- */
    mainSpeakerWidget.value = node.properties.mainSpeaker;
    mainSpeakerWidget.callback = v => {
        node.properties.mainSpeaker = v;
        syncState();
    };

    /* -------------------------------------------------
     * INTERNAL HELPERS
     * ------------------------------------------------- */
    function clearDynamicWidgets() {
        node.widgets = node.widgets.filter(w => !w._isExtraSpeaker && !w._isAddButton);
    }

    function syncState() {
        jsonWidget.value = JSON.stringify({
            speakers: [
                { voice: node.properties.mainSpeaker },
                ...node.properties.extraSpeakers.map(v => ({ voice: v }))
            ]
        });
        node.setDirty(true);
    }

    /* -------------------------------------------------
     * REBUILD SPEAKER LIST
     * ------------------------------------------------- */
    function rebuildUI() {
        clearDynamicWidgets();

        // Speaker 2+
        node.properties.extraSpeakers.forEach((voice, index) => {
            const combo = node.addWidget(
                "combo",
                `speaker_${index + 2}`,
                voice,
                v => {
                    node.properties.extraSpeakers[index] = v;
                    syncState();
                },
                { values: voiceOptions }
            );
            combo._isExtraSpeaker = true;

            const removeBtn = node.addWidget(
                "button",
                "🗑",
                null,
                () => {
                    node.properties.extraSpeakers.splice(index, 1);
                    rebuildUI();
                    syncState();
                }
            );
            removeBtn._isExtraSpeaker = true;
            removeBtn.options = { size: [28, 20] };
        });

        // + Add Speaker (always LAST)
        const addBtn = node.addWidget(
            "button",
            "+ Add Speaker",
            null,
            () => {
                const def = voiceOptions[0] || "";
                node.properties.extraSpeakers.push(def);
                rebuildUI();
                syncState();
            }
        );
        addBtn._isAddButton = true;
    }

    /* -------------------------------------------------
     * INITIAL BUILD
     * ------------------------------------------------- */
    rebuildUI();
}

app.registerExtension({
    name: "PromptSceneRouter.UI",

    nodeCreated(node) {
        try {

            // 🚨 FIRST LINE — do NOT move this
            if (node.comfyClass === "QwenTTSSpeakerBankNode") {
                // Hide backend-only widgets
                //hideWidget(node, "saved_voice_names");
                hideWidget(node, "speakers_json");

                // Preserve any existing onConfigure
                const originalConfigure = node.onConfigure;

                node.onConfigure = function () {
                    if (originalConfigure) {
                        originalConfigure.apply(this, arguments);
                    }

                    // 🔑 SAFE: widgets_values + widget.value are now correct
                    buildQwenSpeakerBankUI(this);
                };
            }

            // ----------------------------------
            // EXISTING PromptSceneRouter logic
            // (leave it exactly as-is below)
            // ----------------------------------

        } catch (e) {
            console.error(
                "[PromptSceneRouter] nodeCreated error on node:",
                node?.comfyClass,
                e
            );
        }

        if (node.comfyClass !== "PromptSceneRouter") return;

        /* =====================================================
         * STATE
         * ===================================================== */
        let isConfigured = false;

        // Restore-safe scene container
        if (!node._scenes) {
            node._scenes = [
                { prompt: "", source: "new", enable_loras: [] }
            ];
        }
        
        function buildSourceOptions(index) {
            const options = ["new"];

            for (let i = 0; i < index; i++) {
                options.push(`${i + 1}st scene`);
            }

            return options;
        }

        /* =====================================================
         * UTILITIES
         * ===================================================== */
        function getSceneDataInput() {
            return node.inputs?.find(i => i.name === "scene_data") ?? null;
        }

        function connectedLoRACount() {
            return node.inputs.filter(
                i => i.name.startsWith("lora_") && i.link !== null
            ).length;
        }

        function persistScenes() {
            node.properties = node.properties || {};
            node.properties._prompt_scene_router = {
                scenes: node._scenes
            };
        }

        function normalizeLoRAs() {
            const count = connectedLoRACount();
            node._scenes.forEach(scene => {
                scene.enable_loras = scene.enable_loras.filter(i => i < count);
            });
        }

        function syncSceneData() {
            if (!isConfigured) return;

            persistScenes();

            const input = getSceneDataInput();
            if (!input) return;

            input.value = JSON.stringify({ scenes: node._scenes });
        }

        /* =====================================================
         * LoRA SOCKET VISIBILITY (MODEL inputs)
         * ===================================================== */
        function updateLoRASockets() {
            const loras = node.inputs.filter(i => i.name.startsWith("lora_"));

            for (let i = 0; i < loras.length; i++) {
                const show = (i === 0) || (loras[i - 1]?.link !== null);
                loras[i].display = show;
            }

            // Force layout refresh (MANDATORY)
            node.size[1] = node.computeSize()[1];
            node.setDirtyCanvas(true, true);
        }
        
        function addMultiline(node, label, initialValue, onChange, rows = 6) {
          const el = document.createElement("textarea");
          el.className = "comfy-multiline"; // optional, but matches common comfy styling
          el.rows = rows;
          el.value = initialValue ?? "";

          el.addEventListener("input", () => onChange(el.value));

          // addDOMWidget(name, type, domElement, options)
          const w = node.addDOMWidget(label, "STRING", el, {
            getValue: () => el.value,
            setValue: (v) => { el.value = v ?? ""; },
          });

          // optional: match node height to content
          node.setSize([node.size[0], node.computeSize()[1]]);
          return w;
        }
        
        /* =====================================================
         * UI RENDERER
         * ===================================================== */
        function render() {
            if (!isConfigured) return;

            // Remove old scene widgets
            node.widgets = node.widgets.filter(w => !w._sceneWidget);

            node._scenes.forEach((scene, index) => {
                const prompt = addMultiline(
                  node,
                  `Scene ${index + 1} Prompt`,
                  scene.prompt,
                  (v) => { scene.prompt = v; syncSceneData(); },
                  6
                );
                prompt._sceneWidget = true;

                const sourceOpts = ["new"];
                for (let i = 0; i < index; i++) {
                    sourceOpts.push(`${i + 1}st scene`);
                }
                sourceOpts.push("prev");

                const source = node.addWidget(
                    "combo",
                    "Source",
                    scene.source,
                    v => {
                        scene.source = v;
                        syncSceneData();
                    },
                    { values: buildSourceOptions(index) }
                );
                source._sceneWidget = true;

                const loraCount = connectedLoRACount();
                for (let i = 0; i < loraCount; i++) {
                    const toggle = node.addWidget(
                        "toggle",
                        `Use LoRA ${i + 1}`,
                        scene.enable_loras.includes(i),
                        v => {
                            if (v && !scene.enable_loras.includes(i))
                                scene.enable_loras.push(i);
                            if (!v)
                                scene.enable_loras = scene.enable_loras.filter(x => x !== i);
                            syncSceneData();
                        }
                    );
                    toggle._sceneWidget = true;
                }
            });

            const addBtn = node.addWidget(
                "button",
                "➕ Add Scene",
                null,
                () => {
                    node._scenes.push({
                        prompt: "", 
                        source: "new",
                        enable_loras: []
                    });
                    normalizeLoRAs();
                    render();
                    syncSceneData();
                }
            );
            addBtn._sceneWidget = true;

            node.setDirtyCanvas(true, true);
        }

        /* =====================================================
         * CONNECTION HANDLER
         * ===================================================== */
        node.onConnectionsChange = () => {
            //updateLoRASockets();
            normalizeLoRAs();
            render();
            syncSceneData();
        };

        /* =====================================================
         * SERIALIZATION (EXPORT / IMPORT)
         * ===================================================== */
        node.onSerialize = function (data) {
            data.properties = data.properties || {};
            data.properties._prompt_scene_router = {
                scenes: node._scenes
            };
        };

        node.onDeserialize = function (data) {
            if (data?.properties?._prompt_scene_router?.scenes) {
                node._scenes = data.properties._prompt_scene_router.scenes;
            }
        };

        /* =====================================================
         * CONFIGURE (PAGE REFRESH SAFE)
         * ===================================================== */
        const origConfigure = node.configure;
        node.configure = function (info) {
            origConfigure?.call(this, info);

            // PRIMARY restore path (refresh-safe)
            if (node.properties?._prompt_scene_router?.scenes) {
                node._scenes = node.properties._prompt_scene_router.scenes;
            }
            // Fallback (execution mirror)
            else if (info?.scene_data) {
                try {
                    const parsed = JSON.parse(info.scene_data);
                    if (parsed?.scenes?.length) {
                        node._scenes = parsed.scenes;
                    }
                } catch {}
            }

            // Guaranteed default
            if (!node._scenes || !node._scenes.length) {
                node._scenes = [
                    { prompt: "", source: "new", enable_loras: [] }
                ];
            }

            isConfigured = true;

            setTimeout(() => {
                //updateLoRASockets();
                normalizeLoRAs();
                render();
                syncSceneData();

                const sd = getSceneDataInput();
                if (sd) sd.display = false;
            }, 0);
        };
    }
});

