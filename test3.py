"""
tsne_weather_domains_clean_clusters.py

Goal:
- Simulate 4 weather-domain feature clouds (Clear, Rain, Snow, Fog)
- Make them mostly separated with a small controlled overlap region
- Visualize with openTSNE (preferred) or sklearn TSNE fallback
- Save figure to file and show it

Usage:
    python tsne_weather_domains_clean_clusters.py
Dependencies:
    numpy, matplotlib, sklearn (optional openTSNE)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

np.random.seed(42)

# -----------------------
# Parameters you can tweak
# -----------------------
n_domains = 4
domains = ["Clear", "Rain", "Snow", "Fog"]
n_per_domain = 350           # samples per domain
feat_dim = 128               # high-dim feature dimension (e.g., deep features)
cluster_separation = 6.0     # larger -> clusters farther apart -> less overlap
intra_scale = 0.6            # smaller -> tighter clusters -> less overlap
mix_rate = 0.04              # fraction of samples to be replaced by cross-domain samples (controls small overlap)
perplexity = 40
random_state = 42
save_path = "tsne_weather_domains.png"

# Colors & markers (pleasant palette)
colors = {
    "Clear": "#2b8cbe",
    "Rain":  "#4daf4a",
    "Snow":  "#fdae61",
    "Fog":   "#984ea3"
}
markers = {
    "Clear": "o",
    "Rain":  "s",
    "Snow":  "^",
    "Fog":   "X"
}

# -----------------------
# Synthesize high-dim features
# -----------------------
# Create well-separated cluster centers in feat_dim via orthogonal-ish vectors
# We'll create centers by sampling gaussian vectors then orthonormalizing (QR) and scaling
raw = np.random.normal(size=(n_domains, feat_dim))
q, _ = np.linalg.qr(raw.T)   # q shape (feat_dim, n_domains)
centers = (q[:, :n_domains].T) * cluster_separation  # (n_domains, feat_dim)

# Generate samples per domain
X_parts = []
y_parts = []
for i, d in enumerate(domains):
    center = centers[i]
    # Tight multivariate gaussian around center
    samples = np.random.normal(loc=0.0, scale=intra_scale, size=(n_per_domain, feat_dim)) + center
    X_parts.append(samples)
    y_parts += [d] * n_per_domain

X = np.vstack(X_parts)
y = np.array(y_parts)

# -----------------------
# Inject a small amount of cross-domain mixing to create controlled overlap
# -----------------------
n_total = X.shape[0]
n_mixed = int(n_total * mix_rate)
if n_mixed > 0:
    # choose indices to replace with samples from other domains (simulate mixed/overlap)
    mixed_idx = np.random.choice(n_total, size=n_mixed, replace=False)
    for idx in mixed_idx:
        # choose a domain different from the original one -> sample from that other domain's distribution
        orig_domain = y[idx]
        other_domains = [d for d in domains if d != orig_domain]
        chosen = np.random.choice(other_domains)
        chosen_idx = domains.index(chosen)
        new_sample = np.random.normal(loc=0.0, scale=intra_scale, size=(1, feat_dim)) + centers[chosen_idx]
        X[idx] = new_sample
        y[idx] = chosen  # label it as the domain it came from (so overlap visible)

# Standardize features before t-SNE
X = StandardScaler().fit_transform(X)

# -----------------------
# Run t-SNE (openTSNE preferred)
# -----------------------
use_open_tsne = False
try:
    from openTSNE import TSNE as oTSNE
    use_open_tsne = True
except Exception:
    use_open_tsne = False

if use_open_tsne:
    tsne = oTSNE(
        n_components=2,
        perplexity=perplexity,
        metric="euclidean",
        n_jobs=8,
        random_state=random_state,
        initialization="pca",
        verbose=False,
    )
    X_2d = tsne.fit(X)
    method_name = "openTSNE"
else:
    # fallback to sklearn
    try:
        from sklearn.manifold import TSNE as sklTSNE
        tsne = sklTSNE(n_components=2, perplexity=perplexity, learning_rate='auto',
                       init='pca', random_state=random_state)
        X_2d = tsne.fit_transform(X)
        method_name = "sklearn TSNE (fallback)"
    except Exception:
        # Final fallback: PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=random_state)
        X_2d = pca.fit_transform(X)
        method_name = "PCA (final fallback)"

# -----------------------
# Plotting: aesthetic & clear legend (English)
# -----------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12
})

plt.figure(figsize=(10, 9), dpi=150)
for d in domains:
    sel = (y == d)
    plt.scatter(
        X_2d[sel, 0], X_2d[sel, 1],
        s=42,
        marker=markers[d],
        color=colors[d],
        label=d,
        alpha=0.86,
        edgecolor="black",
        linewidth=0.35
    )

# Annotate approximate cluster centers on plot (for readability)
for i, d in enumerate(domains):
    sel = (y == d)
    cx = X_2d[sel, 0].mean()
    cy = X_2d[sel, 1].mean()
    plt.text(cx, cy, d, fontsize=11, fontweight='bold',
             ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))

plt.title("t-SNE — Weather Domain Feature Distribution\n(mostly separated clusters with small overlap)", pad=12)
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
leg = plt.legend(title="Weather Domain", fontsize=10, title_fontsize=11, loc='best')
leg.get_frame().set_alpha(0.9)
plt.grid(alpha=0.22)
plt.tight_layout()

# Save and show
plt.savefig(save_path, dpi=200)
print(f"Saved visualization to '{os.path.abspath(save_path)}'  (method: {method_name})")
plt.show()
