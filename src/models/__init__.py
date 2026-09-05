from functools import partial

from src.models.emirt_wrapper import EMIRT
from src.models.ideal_wrapper import IDEAL
from src.models.ixplore_wrapper import IXPLOREBinarised, IXPLOREModel
from src.models.kernel_pca_wrapper import KernelPCA
from src.models.lsirm_wrapper import LSIRM
from src.models.pca_wrapper import PCALinear, PCALogistic
from src.models.tsne_wrapper import TSNELogistic
from src.models.umap_wrapper import UMAPLogistic
from src.models.vae_wrapper import VAE
# from src.models.wnominate_wrapper import WNOMINATE

IXPLORE = partial(
    IXPLOREModel,
    n_iterations=10,
    prior_variance=0.25,
    kernel_name="linear",
    pca_initialization=True,
    sampling_resolution=200,
    scale_weights=False,
    use_point_estimates=True,
)

IXPLORE_BINARISED = partial(
    IXPLOREBinarised,
    n_iterations=10,
    prior_variance=0.25,
    kernel_name="linear",
    pca_initialization=True,
    sampling_resolution=200,
    scale_weights=False,
    use_point_estimates=True,
)

BASELINE_ALGORITHMS = {
    "pca-linear": PCALinear,
    "pca-logistic": PCALogistic,
    "kernel-pca": KernelPCA,
    "tsne-logistic": TSNELogistic,
    "umap-logistic": UMAPLogistic,
    "vae-2layer": partial(VAE, decoder_type="2-layer", beta=1.0),
    # max_iters=6300: empirical cap from an uncapped EVS sp_0.9 probe (23,502
    # iters to converge; loss within 5% of optimum by iter 6,279). Without the
    # KLD term (beta=0) early stopping rarely triggers, so this bounds runtime.
    # Deviates from verbatim ECML 2024 — must be flagged in the report.
    "vae-logistic": partial(VAE, decoder_type="logistic", beta=0.0, max_iters=6300),
    "ideal": IDEAL,
    "emirt": EMIRT,
    # "wnominate": WNOMINATE,
    "lsirm": LSIRM,
    "ixplore": IXPLORE,
    "ixplore-binarised": IXPLORE_BINARISED,
}

MODEL_CLASSES = {
    "pca-linear": PCALinear,
    "pca-logistic": PCALogistic,
    "kernel-pca": KernelPCA,
    "tsne-logistic": TSNELogistic,
    "umap-logistic": UMAPLogistic,
    "ideal": IDEAL,
    "emirt": EMIRT,
    # "wnominate": WNOMINATE,
    "lsirm": LSIRM,
    "vae-2layer": VAE,
    "vae-logistic": VAE,
    "ixplore": IXPLOREModel,
    "ixplore-binarised": IXPLOREBinarised,
}

ALGORITHM_LABELS = {
    "pca-linear": "Linear PCA",
    "pca-logistic": "Logistic PCA",
    "kernel-pca": "Kernel PCA",
    "tsne-logistic": "t-SNE",
    "umap-logistic": "UMAP",
    "vae-2layer": "VAE (2-layer)",
    "vae-logistic": "VAE (logistic)",
    "ideal": "IDEAL",
    "emirt": "emIRT",
    # "wnominate": "W-NOMINATE",
    "lsirm": "LSIRM",
    "ixplore": "IXPLORE",
    "ixplore-binarised": "IXPLORE$^*$",
}
