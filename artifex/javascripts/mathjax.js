// MathJax configuration for Artifex documentation
// Must be loaded BEFORE MathJax library

const artifexMathJaxVendorRoot = new URL("../assets/vendor/", document.currentScript.src);
const artifexMathJaxUsesExplorer = location.protocol !== "file:";

const makeHostedEquationReferencesCrawlable = () => {
  for (const innerLink of document.querySelectorAll(
    '.arithmatex mjx-math[aria-hidden="true"] > a[data-mjx-href]',
  )) {
    innerLink.setAttribute("href", innerLink.dataset.mjxHref);
    // MathJax 4.1's external speech node owns keyboard and screen-reader
    // interaction. Keep the visual-tree anchor out of sequential focus while
    // supplying the standard SVG/HTML destination expected by crawlers.
    innerLink.tabIndex = -1;
  }
};

const makeOfflineEquationReferencesAccessible = () => {
  for (const innerLink of document.querySelectorAll(
    '.arithmatex mjx-math[aria-hidden="true"] > a[href^="#eq"]',
  )) {
    const container = innerLink.closest("mjx-container");
    if (!container || container.parentElement?.classList.contains("artifex-equation-reference")) {
      continue;
    }
    const href = innerLink.getAttribute("href");
    const spokenLabel = container.querySelector("mjx-assistive-mml")?.textContent
      ?.replace(/\s+/g, " ")
      .trim();
    const link = document.createElement("a");
    link.className = "artifex-equation-reference";
    link.href = href;
    link.setAttribute("aria-label", `Go to equation ${spokenLabel || href.slice(1)}`);

    // MathJax's direct-file fallback puts the native anchor inside its
    // aria-hidden visual tree. Move that link contract to ordinary HTML while
    // preserving the exact MathJax glyph subtree and assistive MathML.
    innerLink.replaceWith(...innerLink.childNodes);
    container.removeAttribute("tabindex");
    container.before(link);
    link.append(container);
  }
};

window.MathJax = {
  loader: {
    load: [
      '[tex]/tagformat',
      ...(artifexMathJaxUsesExplorer ? ['ui/lazy'] : []),
    ],
    paths: {
      mathmaps: new URL("mathjax/sre/mathmaps", artifexMathJaxVendorRoot).href,
      sre: new URL("mathjax/sre", artifexMathJaxVendorRoot).href,
    }
  },
  output: {
    font: "mathjax-newcm",
    fontPath: new URL("mathjax-newcm", artifexMathJaxVendorRoot).href,
  },
  options: {
    // MathJax 4's speech explorer uses a worker. Browsers block blob workers
    // in direct file mode, so the offline edition uses MathJax's supported
    // assistive-MathML mode instead. Hosted pages keep the richer explorer.
    enableExplorer: artifexMathJaxUsesExplorer,
    ignoreHtmlClass: ".*|",
    ...(artifexMathJaxUsesExplorer ? {
      lazyMargin: '400px',
      lazyAlwaysTypeset: null,
    } : {}),
    menuOptions: {
      settings: {
        assistiveMml: !artifexMathJaxUsesExplorer,
        braille: artifexMathJaxUsesExplorer,
        speech: artifexMathJaxUsesExplorer,
      },
    },
    processHtmlClass: "arithmatex"
  },
  startup: {
    pageReady: () => {
      // The lazy extension requires actual containers, not selector strings.
      // Resolve them only after the page DOM is complete, immediately before
      // the initial typesetting pass creates semantic equation targets.
      const referenceTargets = [...document.querySelectorAll('.arithmatex')]
        .filter((element) =>
          element.closest('.math-reference-target') || element.textContent.includes('\\eqref{')
        );
      if (artifexMathJaxUsesExplorer) {
        MathJax.startup.document.options.lazyAlwaysTypeset =
          referenceTargets.length ? referenceTargets : null;
      }
      return MathJax.startup.defaultPageReady().then(() => {
        if (artifexMathJaxUsesExplorer) makeHostedEquationReferencesCrawlable();
        else makeOfflineEquationReferencesAccessible();
      });
    }
  },
  tex: {
    packages: {'[+]': ['tagformat']},
    inlineMath: [['\\(', '\\)'], ['$', '$']],
    displayMath: [['\\[', '\\]'], ['$$', '$$']],
    processEscapes: true,
    processEnvironments: true,
    processRefs: true,
    tags: 'ams',
    useLabelIds: true,
    tagformat: {
      // Keep semantic book labels addressable as literal URL fragments.
      id: (id) => id.replace(/\s/g, '_')
    },
    // Generative Models and Deep Learning specific macros
    macros: {
      // Common mathematical notation
      "R": "\\mathbb{R}",
      "C": "\\mathbb{C}",
      "N": "\\mathbb{N}",
      "Z": "\\mathbb{Z}",
      "Q": "\\mathbb{Q}",

      // Vector and matrix notation
      "vec": ["\\mathbf{#1}", 1],
      "mat": ["\\mathbf{#1}", 1],
      "norm": ["\\left\\|#1\\right\\|", 1],
      "abs": ["\\left|#1\\right|", 1],

      // Statistical notation
      "mean": ["\\bar{#1}", 1],
      "var": ["\\text{Var}(#1)", 1],
      "std": ["\\text{Std}(#1)", 1],
      "cov": ["\\text{Cov}(#1, #2)", 2],
      "corr": ["\\text{Corr}(#1, #2)", 2],

      // Probability notation
      "prob": ["\\text{P}(#1)", 1],
      "expect": ["\\mathbb{E}[#1]", 1],
      "given": "\\mid",

      // Machine learning notation
      "NN": "\\mathcal{N}",
      "loss": "\\mathcal{L}",
      "params": "\\theta",
      "weights": "\\mathbf{W}",
      "bias": "\\mathbf{b}",
      "activation": "\\sigma",
      "softmax": "\\text{softmax}",
      "relu": "\\text{ReLU}",
      "gelu": "\\text{GELU}",
      "sigmoid": "\\text{sigmoid}",
      "tanh": "\\text{tanh}",
      "silu": "\\text{SiLU}",
      "swish": "\\text{Swish}",

      // VAE notation
      "vae": "\\text{VAE}",
      "encoder": "q_{\\phi}",
      "decoder": "p_{\\theta}",
      "latent": "\\mathbf{z}",
      "posterior": "q_{\\phi}(\\mathbf{z}|\\mathbf{x})",
      "prior": "p(\\mathbf{z})",
      "elbo": "\\mathcal{L}_{\\text{ELBO}}",
      "kldiv": "\\text{KL}",
      "recon": "\\mathcal{L}_{\\text{recon}}",

      // GAN notation
      "gan": "\\text{GAN}",
      "generator": "G_{\\theta}",
      "discriminator": "D_{\\phi}",
      "adversarial": "\\mathcal{L}_{\\text{adv}}",
      "wgan": "\\text{WGAN}",
      "stylegan": "\\text{StyleGAN}",
      "patchgan": "\\text{PatchGAN}",

      // Diffusion models notation
      "diffusion": "\\text{Diffusion}",
      "ddpm": "\\text{DDPM}",
      "ddim": "\\text{DDIM}",
      "score": "\\mathbf{s}_{\\theta}",
      "noise": "\\boldsymbol{\\epsilon}",
      "timestep": "t",
      "betaschedule": "\\beta_t",
      "alphabar": "\\bar{\\alpha}_t",
      "denoise": "\\mathbf{x}_0",
      "sde": "\\text{SDE}",
      "ode": "\\text{ODE}",

      // Flow models notation
      "flow": "\\text{Flow}",
      "normalizing": "\\text{NormalizingFlow}",
      "realvp": "\\text{RealNVP}",
      "glow": "\\text{Glow}",
      "maf": "\\text{MAF}",
      "iaf": "\\text{IAF}",
      "jacobian": "\\mathbf{J}",
      "logdet": "\\log|\\det \\mathbf{J}|",

      // Energy-based models
      "ebm": "\\text{EBM}",
      "energy": "E_{\\theta}",
      "partition": "Z_{\\theta}",
      "contrastive": "\\mathcal{L}_{\\text{CD}}",
      "langevin": "\\text{Langevin}",
      "mcmc": "\\text{MCMC}",
      "hmc": "\\text{HMC}",
      "nuts": "\\text{NUTS}",

      // Autoregressive models
      "autoregressive": "\\text{AR}",
      "pixelcnn": "\\text{PixelCNN}",
      "wavenet": "\\text{WaveNet}",
      "transformer": "\\text{Transformer}",
      "gpt": "\\text{GPT}",

      // Attention and transformer notation
      "attn": "\\text{Attn}",
      "query": "\\mathbf{Q}",
      "key": "\\mathbf{K}",
      "value": "\\mathbf{V}",
      "multihead": "\\text{MultiHead}",
      "selfattn": "\\text{SelfAttn}",
      "crossattn": "\\text{CrossAttn}",
      "dit": "\\text{DiT}",

      // Protein modeling notation
      "protein": "\\mathbf{P}",
      "residue": "\\mathbf{r}",
      "backbone": "\\mathbf{b}",
      "sidechain": "\\mathbf{s}",
      "coords": "\\mathbf{c}",
      "distances": "\\mathbf{D}",
      "angles": "\\boldsymbol{\\theta}",
      "dihedrals": "\\boldsymbol{\\phi}, \\boldsymbol{\\psi}",

      // Geometric/3D notation
      "point": "\\mathbf{p}",
      "pointcloud": "\\mathcal{P}",
      "mesh": "\\mathcal{M}",
      "surface": "\\mathcal{S}",
      "voxel": "\\mathbf{v}",
      "rotation": "\\mathbf{R}",
      "translation": "\\mathbf{t}",
      "se3": "\\text{SE}(3)",
      "so3": "\\text{SO}(3)",

      // Data notation
      "data": "\\mathbf{x}",
      "sample": "\\mathbf{x}",
      "batch": "\\mathcal{B}",
      "batchsize": "B",
      "dataset": "\\mathcal{D}",
      "train": "\\mathcal{D}_{\\text{train}}",
      "val": "\\mathcal{D}_{\\text{val}}",
      "test": "\\mathcal{D}_{\\text{test}}",

      // Image/vision notation
      "image": "\\mathbf{I}",
      "feature": "\\mathbf{h}",
      "embedding": "\\mathbf{z}",
      "patch": "\\mathbf{p}",
      "channel": "C",
      "height": "H",
      "width": "W",

      // Text/sequence notation
      "sequence": "\\mathbf{x}_{1:T}",
      "token": "x_t",
      "vocab": "\\mathcal{V}",
      "context": "\\mathbf{c}",

      // Audio notation
      "audio": "\\mathbf{a}",
      "waveform": "\\mathbf{w}",
      "spectrogram": "\\mathbf{S}",
      "mel": "\\mathbf{M}",
      "stft": "\\text{STFT}",

      // Optimization notation
      "lr": "\\eta",
      "grad": "\\nabla",
      "update": "\\leftarrow",
      "adam": "\\text{Adam}",
      "sgd": "\\text{SGD}",
      "adamw": "\\text{AdamW}",
      "momentum": "m",
      "epsilon": "\\varepsilon",

      // Loss functions
      "mse": "\\text{MSE}",
      "mae": "\\text{MAE}",
      "crossentropy": "\\text{CE}",
      "bce": "\\text{BCE}",
      "nll": "\\text{NLL}",
      "hinge": "\\text{Hinge}",
      "perceptual": "\\mathcal{L}_{\\text{perceptual}}",

      // Evaluation metrics
      "fid": "\\text{FID}",
      "is": "\\text{IS}",
      "kid": "\\text{KID}",
      "precision": "\\text{Precision}",
      "recall": "\\text{Recall}",
      "lpips": "\\text{LPIPS}",
      "ssim": "\\text{SSIM}",
      "psnr": "\\text{PSNR}",
      "perplexity": "\\text{PPL}",
      "bleu": "\\text{BLEU}",

      // Conditioning notation
      "condition": "\\mathbf{y}",
      "class": "y",
      "guidance": "w",
      "cfg": "\\text{CFG}",
      "unconditional": "\\emptyset",

      // Sampling notation
      "sample": "\\mathbf{x} \\sim p_{\\theta}",
      "ancestral": "\\text{Ancestral}",
      "greedy": "\\text{Greedy}",
      "beam": "\\text{Beam}",
      "nucleus": "\\text{Nucleus}",
      "topk": "\\text{Top-}k",
      "temperature": "\\tau",

      // JAX/Flax specific
      "jax": "\\text{JAX}",
      "flax": "\\text{Flax}",
      "nnx": "\\text{NNX}",
      "jit": "\\text{JIT}",
      "vmap": "\\text{vmap}",
      "pmap": "\\text{pmap}",
      "scan": "\\text{scan}",
      "pytree": "\\text{PyTree}",

      // Distribution notation
      "gaussian": "\\mathcal{N}",
      "uniform": "\\mathcal{U}",
      "bernoulli": "\\text{Bernoulli}",
      "categorical": "\\text{Categorical}",
      "dirichlet": "\\text{Dirichlet}",

      // Architecture notation
      "unet": "\\text{U-Net}",
      "resnet": "\\text{ResNet}",
      "vit": "\\text{ViT}",
      "cnn": "\\text{CNN}",
      "rnn": "\\text{RNN}",
      "lstm": "\\text{LSTM}",
      "gru": "\\text{GRU}",

      // Regularization
      "dropout": "\\text{Dropout}",
      "batchnorm": "\\text{BatchNorm}",
      "layernorm": "\\text{LayerNorm}",
      "groupnorm": "\\text{GroupNorm}",
      "spectral": "\\text{SpectralNorm}",

      // Performance metrics
      "flops": "\\text{FLOPs}",
      "throughput": "\\text{Throughput}",
      "latency": "\\text{Latency}",
      "memory": "\\text{Memory}",

      // Set notation
      "argmax": "\\operatorname*{arg\\,max}",
      "argmin": "\\operatorname*{arg\\,min}"
    }
  }
};
