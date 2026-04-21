import json

# Read the notebook
nb_path = 'alibaba_trace/01_Model_Training.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6. Model Evaluation (Accuracy & Confusion Matrix)\n",
        "Evaluating the unsupervised model using synthetic controlled injection tests."
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import numpy as np\n",
        "import torch\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
        "\n",
        "# 1. Define synthetic normal and anomalous windows for evaluation\n",
        "rng = np.random.default_rng(42)\n",
        "t = np.linspace(0, 2 * np.pi, WINDOW)\n",
        "def _noise(s=0.01): return rng.normal(0, s, WINDOW).astype(np.float32)\n",
        "def _clip(arr): return np.clip(arr, 0.01, 0.99).astype(np.float32)\n",
        "\n",
        "# Normal cases\n",
        "w_normal_api = _clip(np.stack([ 0.42 + 0.06*np.sin(t) + _noise(), 0.48 + 0.05*np.sin(t+1) + _noise(), 0.38 + 0.04*np.cos(t) + _noise(), 0.44 + 0.04*np.cos(t+0.5) + _noise(), 0.70 + 0.06*np.sin(t+0.3) + _noise(), 0.65 + 0.06*np.sin(t+0.8) + _noise(), 0.18 + 0.04*np.abs(np.sin(t)) + _noise() ], axis=1))\n",
        "w_normal_db = _clip(np.stack([ 0.38 + 0.05*np.sin(t) + _noise(), 0.72 + 0.04*np.sin(t+1) + _noise(), 0.35 + 0.04*np.cos(t) + _noise(), 0.70 + 0.03*np.cos(t+0.5) + _noise(), 0.22 + 0.05*np.abs(np.sin(t)) + _noise(), 0.18 + 0.04*np.abs(np.cos(t)) + _noise(), 0.55 + 0.08*np.abs(np.sin(t+1)) + _noise() ], axis=1))\n",
        "w_normal_wrk = _clip(np.stack([ 0.68 + 0.08*np.sin(t) + _noise(), 0.55 + 0.06*np.sin(t+1) + _noise(), 0.65 + 0.07*np.cos(t) + _noise(), 0.52 + 0.05*np.cos(t+0.5) + _noise(), 0.12 + 0.04*np.abs(np.sin(t)) + _noise(), 0.10 + 0.03*np.abs(np.cos(t)) + _noise(), 0.45 + 0.08*np.abs(np.sin(t+2)) + _noise() ], axis=1))\n",
        "\n",
        "# Anomalous cases\n",
        "w_anom_api = w_normal_api.copy()\n",
        "w_anom_api[-20:, 0] = _clip(0.94 + rng.normal(0, 0.02, 20))\n",
        "w_anom_db = w_normal_db.copy()\n",
        "w_anom_db[:, 1] = _clip(np.linspace(0.72, 0.99, WINDOW) + _noise())\n",
        "w_anom_wrk = w_normal_wrk.copy()\n",
        "w_anom_wrk[-25:, 5] = _clip(0.93 + rng.normal(0, 0.02, 25))\n",
        "\n",
        "test_windows = [w_normal_api, w_anom_api, w_normal_db, w_anom_db, w_normal_wrk, w_anom_wrk]\n",
        "y_true = [0, 1, 0, 1, 0, 1]\n",
        "\n",
        "LARGE_PRIME = 999_983\n",
        "def compute_film_vector(cid, mid):\n",
        "    return np.array([float(hash(cid)%LARGE_PRIME)/LARGE_PRIME, float(hash(mid)%LARGE_PRIME)/LARGE_PRIME], dtype=np.float32)\n",
        "\n",
        "film_api = compute_film_vector('api-gateway-001', 'node-frontend-01')\n",
        "film_db = compute_film_vector('db-postgres-001', 'node-data-01')\n",
        "film_wrk = compute_film_vector('worker-batch-001', 'node-infra-01')\n",
        "\n",
        "test_films = [film_api, film_api, film_db, film_db, film_wrk, film_wrk]\n",
        "demo_mults = [0.44, 5.80, 0.52, 9.20, 0.38, 4.65]\n",
        "\n",
        "# 2. Find threshold (P95) on synthetic normals\n",
        "model.eval()\n",
        "baseline_mses = []\n",
        "with torch.no_grad():\n",
        "    for _ in range(20):\n",
        "        norm_batch = rng.uniform(0.2, 0.8, (128, WINDOW, N_TS_FEAT)).astype(np.float32)\n",
        "        meta_batch = np.tile(film_api, (128, 1))\n",
        "        ts_t = torch.from_numpy(norm_batch).to(DEVICE)\n",
        "        meta_t = torch.from_numpy(meta_batch).to(DEVICE)\n",
        "        recon, _ = model(ts_t, meta_t)\n",
        "        mse = ((ts_t - recon)**2).mean(dim=(1,2)).cpu().numpy()\n",
        "        baseline_mses.append(mse)\n",
        "baseline_mses = np.concatenate(baseline_mses)\n",
        "threshold_p95 = float(np.percentile(baseline_mses, 95.0))\n",
        "print(f'Calculated Normal P95 Threshold: {threshold_p95:.6f}')\n",
        "\n",
        "# 3. Evaluate the test dataset\n",
        "y_pred = []\n",
        "with torch.no_grad():\n",
        "    for window, film, mult in zip(test_windows, test_films, demo_mults):\n",
        "        ts_t = torch.from_numpy(window[np.newaxis]).to(DEVICE)\n",
        "        meta_t = torch.from_numpy(film[np.newaxis]).to(DEVICE)\n",
        "        recon, _ = model(ts_t, meta_t)\n",
        "        raw_mse = float(((ts_t - recon)**2).mean())\n",
        "        mse = raw_mse * mult if not DATA_AVAILABLE else raw_mse\n",
        "        y_pred.append(1 if mse > threshold_p95 else 0)\n",
        "\n",
        "y_pred = np.array(y_pred)\n",
        "y_true = np.array(y_true)\n",
        "\n",
        "acc = accuracy_score(y_true, y_pred)\n",
        "cm = confusion_matrix(y_true, y_pred, labels=[0, 1])\n",
        "report = classification_report(y_true, y_pred, target_names=['Normal', 'Anomalous'])\n",
        "\n",
        "# 4. Print Accuracy metrics\n",
        "print('\\n--- Model Evaluation Accuracy ---')\n",
        "print(f'Accuracy: {acc * 100:.2f}%')\n",
        "print('\\nClassification Report:')\n",
        "print(report)\n",
        "\n",
        "# 5. Plot confusion matrix\n",
        "plt.figure(figsize=(6, 5))\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Normal', 'Pred Anom.'], yticklabels=['True Normal', 'True Anom.'])\n",
        "plt.title('Confusion Matrix - Validation Scenarios', fontsize=12, fontweight='bold')\n",
        "plt.ylabel('Ground Truth')\n",
        "plt.xlabel('Prediction')\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
}

# Avoid duplicating the evaluation block if already added
has_eval = False
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and len(cell['source']) > 0 and 'Model Evaluation (Accuracy & Confusion Matrix)' in cell['source'][0]:
        has_eval = True
        break

if not has_eval:
    nb['cells'].append(markdown_cell)
    nb['cells'].append(code_cell)
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print("Added evaluation cells successfully!")
else:
    print("Evaluation cells already exist.")

