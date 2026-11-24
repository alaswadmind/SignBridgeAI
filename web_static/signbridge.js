(() => {
  const $ = (id) => document.getElementById(id);
  const video = $("cameraVideo");
  const overlay = $("landmarkOverlay");
  const ctx = overlay.getContext("2d");
  const stage = $("cameraStage");
  const connectionPairs = [
    [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],[0,17]
  ];

  let stream = null;
  let timer = null;
  let busy = false;
  let lastResult = null;
  let recording = false;
  let capturedRows = [];
  let frameNo = 0;
  let toastTimer = null;

  function toast(message) {
    const node = $("toast");
    node.textContent = message;
    node.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => node.classList.remove("show"), 2600);
  }

  function setCameraStatus(message, state = "") {
    const status = $("cameraStatus");
    status.classList.remove("running", "error");
    if (state) status.classList.add(state);
    $("cameraStatusText").textContent = message;
  }

  function clearOverlay() {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    $("metricLandmarks").textContent = "— / 21";
    $("metricFeatures").textContent = "— / 89";
  }

  function drawLandmarks(points) {
    const width = overlay.width;
    const height = overlay.height;
    ctx.clearRect(0, 0, width, height);
    ctx.lineWidth = Math.max(2, width / 360);
    ctx.strokeStyle = "rgba(111, 226, 211, .76)";
    ctx.shadowColor = "rgba(111, 226, 211, .55)";
    ctx.shadowBlur = 7;
    for (const [a, b] of connectionPairs) {
      ctx.beginPath();
      ctx.moveTo(points[a][0] * width, points[a][1] * height);
      ctx.lineTo(points[b][0] * width, points[b][1] * height);
      ctx.stroke();
    }
    ctx.shadowBlur = 9;
    points.forEach((point, index) => {
      ctx.beginPath();
      ctx.fillStyle = index === 0 || [4, 8, 12, 16, 20].includes(index) ? "#c9aaff" : "#8ff2df";
      ctx.arc(point[0] * width, point[1] * height, Math.max(2.4, width / 230), 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.shadowBlur = 0;
  }

  function updatePrediction(result, latencyMs) {
    const prediction = result.prediction;
    if (!prediction) return;
    lastResult = result;
    const letter = $("predictionLetter");
    const old = letter.textContent;
    letter.textContent = prediction.arabic;
    if (old !== prediction.arabic) {
      letter.classList.remove("pop");
      void letter.offsetWidth;
      letter.classList.add("pop");
      setTimeout(() => letter.classList.remove("pop"), 220);
    }
    $("predictionName").textContent = prediction.name;
    $("classId").textContent = prediction.class_id;
    const confidence = Math.max(0, Math.min(1, prediction.confidence));
    $("confidenceValue").textContent = `${(confidence * 100).toFixed(1)}%`;
    $("confidenceBar").style.width = `${confidence * 100}%`;
    $("appendLetter").disabled = false;
    $("metricLandmarks").textContent = `${result.landmarks.length} / 21`;
    $("metricFeatures").textContent = `${result.feature_count} / 89`;
    $("metricLatency").textContent = `${latencyMs} ms`;
    $("frameLabel").textContent = "تم استخراج 89 ميزة";

    const candidates = $("candidateList");
    candidates.replaceChildren();
    result.candidates.forEach((item) => {
      const entry = document.createElement("div");
      entry.className = "candidate";
      const glyph = document.createElement("span");
      glyph.className = "cand-letter";
      glyph.textContent = item.arabic;
      const score = document.createElement("span");
      score.className = "cand-score";
      score.textContent = `${(item.confidence * 100).toFixed(1)}%`;
      entry.append(glyph, score);
      candidates.append(entry);
    });
    drawLandmarks(result.landmarks);

    if (recording) {
      frameNo += 1;
      capturedRows.push({
        features: result.features,
        label: $("recordLabel").value,
        imageName: `webcam_frame_${String(frameNo).padStart(6, "0")}`,
      });
      $("recordCount").textContent = `${capturedRows.length} إطار مسجل`;
    }
  }

  async function processFrame() {
    if (busy || !stream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    busy = true;
    const startedAt = performance.now();
    try {
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      const image = canvas.toDataURL("image/jpeg", 0.76);
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image }),
      });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || "تعذّر تحليل الإطار");
      if (!result.hand_detected) {
        setCameraStatus("لم تُكتشف يد — قرّب اليد إلى منتصف الإطار", "running");
        $("predictionName").textContent = "في انتظار إشارة واضحة";
        $("frameLabel").textContent = "ابحث عن اليد";
        $("appendLetter").disabled = true;
        $("candidateList").innerHTML = '<div class="candidate-empty">لم تُكتشف يد في هذا الإطار</div>';
        clearOverlay();
        lastResult = null;
        return;
      }
      setCameraStatus("الكاميرا تعمل — الإشارة قيد التحليل", "running");
      updatePrediction(result, Math.round(performance.now() - startedAt));
    } catch (error) {
      setCameraStatus("تعذّر تحليل الإطار", "error");
      $("frameLabel").textContent = "تحقق من اتصال المعالجة";
      console.error(error);
    } finally {
      busy = false;
    }
  }

  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: "user" },
      });
      video.srcObject = stream;
      await video.play();
      stage.classList.add("active");
      $("startCamera").disabled = true;
      $("stopCamera").disabled = false;
      $("recordButton").disabled = false;
      $("liveBadge").classList.add("on");
      setCameraStatus("الكاميرا تعمل — ضع يدك في الإطار", "running");
      timer = setInterval(processFrame, 500);
      processFrame();
    } catch (error) {
      console.error(error);
      setCameraStatus("لم نتمكن من فتح الكاميرا", "error");
      toast(error.name === "NotAllowedError" ? "اسمح للمتصفح باستخدام الكاميرا أولاً." : "تحقق من اتصال الكاميرا ثم حاول مجدداً.");
    }
  }

  function stopCamera() {
    if (timer) clearInterval(timer);
    timer = null;
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null;
    video.srcObject = null;
    stage.classList.remove("active");
    $("startCamera").disabled = false;
    $("stopCamera").disabled = true;
    $("recordButton").disabled = true;
    $("liveBadge").classList.remove("on");
    if (recording) {
      recording = false;
      $("recordButton").classList.remove("recording");
      $("recordButton").innerHTML = '<span class="record-dot"></span> ابدأ تسجيل العينة';
      $("downloadCsv").disabled = capturedRows.length === 0;
      $("recordCount").textContent = `${capturedRows.length} إطار مسجل`;
    }
    setCameraStatus("تم إيقاف الكاميرا");
    $("frameLabel").textContent = "تحليل الإطار";
    $("appendLetter").disabled = true;
    $("predictionLetter").textContent = "—";
    $("predictionName").textContent = "في انتظار إشارة";
    $("confidenceValue").textContent = "—";
    $("confidenceBar").style.width = "0";
    $("candidateList").innerHTML = '<div class="candidate-empty">ستظهر النتائج هنا عند اكتشاف اليد</div>';
    $("classId").textContent = "—";
    $("metricLatency").textContent = "—";
    lastResult = null;
    clearOverlay();
  }

  function appendText(value) {
    const field = $("textOutput");
    const start = field.selectionStart ?? field.value.length;
    const end = field.selectionEnd ?? field.value.length;
    field.value = field.value.slice(0, start) + value + field.value.slice(end);
    const caret = start + value.length;
    field.focus();
    field.setSelectionRange(caret, caret);
    updateCharacterCount();
  }

  function updateCharacterCount() {
    $("characterCount").textContent = Array.from($("textOutput").value).length;
  }

  function downloadFeatures() {
    if (!capturedRows.length) return;
    const headers = [...window.SIGNBRIDGE_FEATURE_COLS, "4190", "4191"];
    const quote = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
    const lines = [headers.map(quote).join(",")];
    capturedRows.forEach((row) => {
      lines.push([...row.features, row.label, row.imageName].map(quote).join(","));
    });
    const blob = new Blob(["\ufeff" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `signbridge_features_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }

  $("startCamera").addEventListener("click", startCamera);
  $("stopCamera").addEventListener("click", stopCamera);
  $("appendLetter").addEventListener("click", () => {
    if (!lastResult) return;
    appendText(lastResult.prediction.arabic);
    toast(`أُضيف الحرف ${lastResult.prediction.arabic}`);
  });
  $("spaceButton").addEventListener("click", () => appendText(" "));
  $("backspaceButton").addEventListener("click", () => {
    const field = $("textOutput");
    const start = field.selectionStart ?? field.value.length;
    const end = field.selectionEnd ?? field.value.length;
    if (start !== end) field.value = field.value.slice(0, start) + field.value.slice(end);
    else if (start > 0) field.value = field.value.slice(0, start - 1) + field.value.slice(end);
    field.focus();
    field.setSelectionRange(Math.max(0, start - 1), Math.max(0, start - 1));
    updateCharacterCount();
  });
  $("clearButton").addEventListener("click", () => { $("textOutput").value = ""; updateCharacterCount(); });
  $("textOutput").addEventListener("input", updateCharacterCount);
  $("copyButton").addEventListener("click", async () => {
    const value = $("textOutput").value;
    if (!value) return toast("مساحة الكتابة فارغة.");
    try { await navigator.clipboard.writeText(value); toast("تم نسخ النص."); }
    catch { $("textOutput").select(); document.execCommand("copy"); toast("تم نسخ النص."); }
  });
  $("recordButton").addEventListener("click", () => {
    recording = !recording;
    const button = $("recordButton");
    button.classList.toggle("recording", recording);
    button.innerHTML = recording ? '<span class="record-dot"></span> إيقاف تسجيل العينة' : '<span class="record-dot"></span> ابدأ تسجيل العينة';
    $("downloadCsv").disabled = recording || capturedRows.length === 0;
    $("recordCount").textContent = recording ? `التسجيل جارٍ · ${capturedRows.length} إطار` : `${capturedRows.length} إطار مسجل`;
    if (recording) toast("بدأ تسجيل المتجهات المكتشفة.");
    else toast(`انتهى التسجيل: ${capturedRows.length} إطار.`);
  });
  $("downloadCsv").addEventListener("click", downloadFeatures);
  $("toggleFeatures").addEventListener("click", () => {
    const preview = $("featurePreview");
    preview.hidden = !preview.hidden;
    if (!preview.hidden) {
      preview.textContent = lastResult
        ? lastResult.features.map((value, i) => `${window.SIGNBRIDGE_FEATURE_COLS[i]}: ${Number(value).toFixed(7)}`).join("\n")
        : "لا توجد ميزات بعد.";
      $("toggleFeatures").textContent = "إخفاء المتجه";
    } else $("toggleFeatures").textContent = "عرض آخر متجه";
  });
  window.addEventListener("beforeunload", () => {
    if (stream) stream.getTracks().forEach((track) => track.stop());
  });

  fetch("/api/health").then((r) => r.json()).then((health) => {
    if (health.status === "ready" && health.features === 89) {
      $("cameraHint").textContent = `النموذج جاهز · ${health.classes} فئة`;
    }
  }).catch(() => setCameraStatus("تعذّر الاتصال بالخادم", "error"));
})();
