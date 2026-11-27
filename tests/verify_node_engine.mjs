import assert from 'node:assert/strict';

const BASE_URL = process.env.DIGITAI_URL || 'http://127.0.0.1:3000';

async function main() {
  const healthRes = await fetch(`${BASE_URL}/api/health`);
  assert.equal(healthRes.status, 200);
  const health = await healthRes.json();
  assert.equal(health.model_loaded, true, 'Expected model_loaded=true');
  assert.equal(health.ready, true, 'Expected ready=true');
  assert.equal(health.model_state, 'ready', 'Expected model_state=ready');
  console.log('Health check OK:', health);

  const wsUrl = BASE_URL.replace(/^http/, 'ws');
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('WebSocket connection timeout')), 5000);
    const ws = new WebSocket(wsUrl, 'vite-hmr');
    ws.addEventListener('open', () => {
      clearTimeout(timer);
      ws.close();
      resolve();
    });
    ws.addEventListener('error', (err) => {
      clearTimeout(timer);
      reject(new Error(`WebSocket error: ${err.message || 'failed to connect'}`));
    });
  });
  console.log('Vite WebSocket connection OK:', wsUrl);

  const infoRes = await fetch(`${BASE_URL}/api/model-info`);
  assert.equal(infoRes.status, 200);
  const info = await infoRes.json();
  assert.ok(info.metrics.enhanced_cnn.test_accuracy >= 0.98, 'Expected >= 98% MNIST test accuracy');
  console.log('Model Info OK | Enhanced Test Acc:', info.metrics.enhanced_cnn.test_accuracy);

  const samplesRes = await fetch(`${BASE_URL}/api/samples`);
  assert.equal(samplesRes.status, 200);
  const { samples } = await samplesRes.json();
  assert.ok(samples.length >= 4, 'Expected sample gallery images');

  for (const s of samples) {
    const predRes = await fetch(`${BASE_URL}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: s.image_base64, mode: s.mode }),
    });
    assert.equal(predRes.status, 200, `Failed predicting sample ${s.id}`);
    const pred = await predRes.json();
    console.log(`Sample ${s.id} (expected ${s.expected}) -> predicted ${pred.predicted_number} (${(pred.confidence * 100).toFixed(2)}%, ${pred.processing_time_ms} ms)`);
    assert.equal(pred.predicted_number, s.expected, `Mismatch on sample ${s.id}`);
  }

  const badRes = await fetch(`${BASE_URL}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode: 'single' }),
  });
  assert.equal(badRes.status, 400);

  const evalRes = await fetch(`${BASE_URL}/api/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ num_samples: 1000 }),
  });
  assert.equal(evalRes.status, 200);
  const evalJson = await evalRes.json();
  assert.ok(evalJson.accuracy >= 0.97);
  console.log('Evaluate OK:', {
    samples: evalJson.samples_evaluated,
    accuracy: evalJson.accuracy,
    f1_macro: evalJson.f1_macro,
  });

  console.log('ALL NODE SERVER INTEGRATION CHECKS PASSED!');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
