(() => {
  const canvas = document.getElementById("draw-canvas");
  const ctx = canvas.getContext("2d");
  const brushRange = document.getElementById("brush-size");
  const brushVal = document.getElementById("brush-val");
  const historyKey = "digitai_flask_history";
  let drawing = false;
  let strokes = [];

  function initCanvas() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    strokes = [ctx.getImageData(0, 0, canvas.width, canvas.height)];
  }
  initCanvas();

  function getPos(evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (evt.clientX - rect.left) * scaleX,
      y: (evt.clientY - rect.top) * scaleY,
    };
  }

  canvas.addEventListener("pointerdown", (e) => {
    drawing = true;
    canvas.setPointerCapture(e.pointerId);
    const p = getPos(e);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = Number(brushRange.value);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    const p = getPos(e);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  });

  canvas.addEventListener("pointerup", () => {
    if (!drawing) return;
    drawing = false;
    strokes.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
  });

  brushRange.addEventListener("input", () => {
    brushVal.textContent = `${brushRange.value}px`;
  });

  document.getElementById("btn-undo").addEventListener("click", () => {
    if (strokes.length > 1) {
      strokes.pop();
      ctx.putImageData(strokes[strokes.length - 1], 0, 0);
    }
  });

  document.getElementById("btn-clear").addEventListener("click", initCanvas);

  function getSelectedMode() {
    return document.querySelector('input[name="mode"]:checked')?.value || "multi";
  }

  function renderProbChart(probs) {
    const c = document.getElementById("prob-chart");
    const cctx = c.getContext("2d");
    c.width = c.clientWidth || 420;
    cctx.clearRect(0, 0, c.width, c.height);
    const barW = (c.width - 40) / 10;
    probs.forEach((p, idx) => {
      const x = 20 + idx * barW + 4;
      const h = Math.max(2, p * 75);
      const y = 88 - h;
      cctx.fillStyle = p === Math.max(...probs) ? "#0d6efd" : "#94a3b8";
      cctx.fillRect(x, y, barW - 8, h);
      cctx.fillStyle = "#334155";
      cctx.font = "11px monospace";
      cctx.fillText(String(idx), x + (barW - 14) / 2, 106);
    });
  }

  function renderResult(data) {
    const errBox = document.getElementById("error-box");
    const warnBox = document.getElementById("warning-box");
    errBox.classList.add("d-none");
    warnBox.classList.add("d-none");

    document.getElementById("pred-number").textContent = data.predicted_number;
    document.getElementById("pred-meta").textContent =
      `Confidence: ${(data.confidence * 100).toFixed(2)}% | Time: ${data.processing_time_ms} ms | Digits: ${data.digit_count}`;

    if (data.warnings && data.warnings.length) {
      warnBox.textContent = data.warnings.join(" ");
      warnBox.classList.remove("d-none");
    }

    const strip = document.getElementById("digits-strip");
    strip.innerHTML = "";
    (data.digits || []).forEach((d) => {
      const div = document.createElement("div");
      div.className = "digit-tile";
      div.innerHTML = `<img src="${d.processed_digit_base64}" alt="Digit ${d.digit}" /><div class="fw-bold font-monospace">${d.digit}</div><div class="small text-muted">${(d.confidence * 100).toFixed(1)}%</div>`;
      strip.appendChild(div);
    });

    if (data.primary_probabilities) {
      renderProbChart(data.primary_probabilities);
    }

    const previewImg = document.getElementById("preview-img");
    if (data.processed_preview_base64) {
      previewImg.src = data.processed_preview_base64;
      previewImg.classList.remove("d-none");
    }

    const hist = JSON.parse(localStorage.getItem(historyKey) || "[]");
    hist.unshift({
      time: new Date().toLocaleTimeString(),
      number: data.predicted_number,
      confidence: (data.confidence * 100).toFixed(1),
      mode: data.mode,
    });
    localStorage.setItem(historyKey, JSON.stringify(hist.slice(0, 15)));
    renderHistory();
  }

  function renderHistory() {
    const hist = JSON.parse(localStorage.getItem(historyKey) || "[]");
    const el = document.getElementById("history-list");
    if (!hist.length) {
      el.textContent = "No predictions yet.";
      return;
    }
    el.innerHTML = hist
      .map(
        (h) =>
          `<div class="d-flex justify-content-between border-bottom py-1"><span class="font-monospace fw-semibold">${h.number}</span><span>${h.mode} · ${h.confidence}% · ${h.time}</span></div>`
      )
      .join("");
  }
  renderHistory();

  document.getElementById("btn-clear-history").addEventListener("click", () => {
    localStorage.removeItem(historyKey);
    renderHistory();
  });

  async function predictBase64(dataUrl) {
    const errBox = document.getElementById("error-box");
    errBox.classList.add("d-none");
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl, mode: getSelectedMode() }),
    });
    const data = await res.json();
    if (!res.ok) {
      errBox.textContent = data.error || "Prediction failed";
      errBox.classList.remove("d-none");
      return;
    }
    renderResult(data);
  }

  document.getElementById("btn-predict-canvas").addEventListener("click", () => {
    predictBase64(canvas.toDataURL("image/png"));
  });

  const fileInput = document.getElementById("file-input");
  const dropZone = document.getElementById("drop-zone");

  async function uploadFile(file) {
    const form = new FormData();
    form.append("image", file);
    form.append("mode", getSelectedMode());
    const errBox = document.getElementById("error-box");
    errBox.classList.add("d-none");
    const res = await fetch("/api/predict", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      errBox.textContent = data.error || "Upload prediction failed";
      errBox.classList.remove("d-none");
      return;
    }
    renderResult(data);
  }

  fileInput.addEventListener("change", (e) => {
    if (e.target.files[0]) uploadFile(e.target.files[0]);
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
  });

  let modelReady = false;
  const predictBtn = document.getElementById("btn-predict-canvas");
  const initBox = document.getElementById("init-status-box");

  function updateModelGate(h, connected = true) {
    modelReady = connected && Boolean(h && (h.ready || h.model_loaded));
    document.getElementById("nav-accuracy").textContent =
      h && h.test_accuracy ? `MNIST Accuracy: ${(h.test_accuracy * 100).toFixed(2)}%` : "MNIST Accuracy: --";
    document.getElementById("nav-status").textContent = !connected
      ? "Status: Reconnecting..."
      : modelReady
      ? "Status: CNN model ready."
      : h && h.model_state === "training"
      ? "Status: Training CNN for the first time."
      : h && h.model_state === "error"
      ? "Status: CNN model initialization failed."
      : "Status: Initializing CNN model.";
    if (predictBtn) {
      predictBtn.disabled = !modelReady;
    }
    if (initBox) {
      if (modelReady) {
        initBox.classList.add("d-none");
      } else {
        initBox.textContent = !connected
          ? "Backend disconnected. Attempting automatic reconnection..."
          : (h && h.message) || "Initializing CNN model.";
        initBox.classList.remove("d-none");
      }
    }
  }

  function loadModelInfo() {
    fetch("/api/model-info")
      .then((r) => r.json())
      .then((info) => {
        const m = info.metrics;
        if (!m) return;
        const enh = m.enhanced_cnn;
        const base = m.baseline_cnn;
        document.getElementById("model-metrics-table").innerHTML = `
          <table class="table table-sm table-bordered font-monospace mb-0">
            <thead><tr><th>Metric</th><th>Baseline</th><th>Enhanced</th></tr></thead>
            <tbody>
              <tr><td>Test Accuracy</td><td>${(base.test_accuracy * 100).toFixed(2)}%</td><td><strong>${(enh.test_accuracy * 100).toFixed(2)}%</strong></td></tr>
              <tr><td>Test Loss</td><td>${base.test_loss.toFixed(4)}</td><td>${enh.test_loss.toFixed(4)}</td></tr>
              <tr><td>Macro F1</td><td>${(base.f1_macro * 100).toFixed(2)}%</td><td>${(enh.f1_macro * 100).toFixed(2)}%</td></tr>
              <tr><td>Parameters</td><td>${base.parameters.toLocaleString()}</td><td>${m.parameters.toLocaleString()}</td></tr>
            </tbody>
          </table>`;
      });
  }

  let infoLoaded = false;
  function pollHealth() {
    fetch("/api/health")
      .then((r) => {
        if (!r.ok) throw new Error("Health check failed");
        return r.json();
      })
      .then((h) => {
        updateModelGate(h, true);
        if (h.ready || h.model_loaded) {
          if (!infoLoaded) {
            infoLoaded = true;
            loadModelInfo();
          }
          setTimeout(pollHealth, 15000);
        } else if (h.model_state !== "error") {
          setTimeout(pollHealth, 2500);
        }
      })
      .catch(() => {
        infoLoaded = false;
        updateModelGate(null, false);
        setTimeout(pollHealth, 3000);
      });
  }

  pollHealth();

  fetch("/api/samples")
    .then((r) => r.json())
    .then((data) => {
      const grid = document.getElementById("samples-grid");
      (data.samples || []).forEach((s) => {
        const btn = document.createElement("button");
        btn.className = "btn btn-outline-secondary btn-sm";
        btn.textContent = s.label;
        btn.addEventListener("click", () => {
          if (modelReady) predictBase64(s.image_base64);
        });
        grid.appendChild(btn);
      });
    });

  const evalBtn = document.getElementById("btn-run-eval");
  if (evalBtn) {
    evalBtn.addEventListener("click", async () => {
      const out = document.getElementById("eval-output");
      out.textContent = "Running...";
      const res = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ num_samples: 1000 }),
      });
      const data = await res.json();
      out.textContent = res.ok
        ? `Acc: ${(data.accuracy * 100).toFixed(2)}% | F1: ${(data.f1_macro * 100).toFixed(2)}% | ${data.ms_per_digit} ms/digit`
        : data.error || "Evaluation failed";
    });
  }
})();
