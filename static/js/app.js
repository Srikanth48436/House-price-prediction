// House Price Predictor – frontend JS

const form      = document.getElementById("predict-form");
const resultPanel = document.getElementById("result-panel");
const errorPanel  = document.getElementById("error-panel");
const submitBtn   = document.getElementById("submit-btn");
const btnText     = document.getElementById("btn-text");

function fmt(n) {
  return "$" + Number(n).toLocaleString("en-US");
}

function showResult(data) {
  document.getElementById("result-price").textContent = fmt(data.predicted_price);
  document.getElementById("result-range").textContent =
    `Confidence range: ${fmt(data.confidence_range[0])} – ${fmt(data.confidence_range[1])}`;
  document.getElementById("result-sub").textContent =
    `Model R² = ${data.model_r2} · Based on Ames Housing Dataset`;

  resultPanel.classList.remove("hidden");
  errorPanel.classList.add("hidden");
  resultPanel.scrollIntoView({ behavior: "smooth", block: "center" });
}

function showError(msg) {
  document.getElementById("error-msg").textContent = "Error: " + msg;
  errorPanel.classList.remove("hidden");
  resultPanel.classList.add("hidden");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Collect form values
  const formData = new FormData(form);
  const payload = {};
  for (const [key, val] of formData.entries()) {
    const parsed = parseFloat(val);
    payload[key] = isNaN(parsed) ? val : parsed;
  }

  // UI: loading state
  submitBtn.disabled = true;
  btnText.textContent = "⏳ Predicting…";
  resultPanel.classList.add("hidden");
  errorPanel.classList.add("hidden");

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.success) {
      showResult(data);
    } else {
      showError(data.error || "Unknown error");
    }
  } catch (err) {
    showError("Network error: " + err.message);
  } finally {
    submitBtn.disabled = false;
    btnText.textContent = "🔮 Predict Price";
  }
});
