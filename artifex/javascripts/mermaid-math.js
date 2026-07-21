(() => {
  const mermaidVersion = "11.16.0";
  const katexVersion = "0.16.45";
  const stylesheetLink = document.getElementById("artifex-katex-stylesheet");
  const shadowCss = window.__artifexKatexShadowCss;
  const mermaid = window.mermaid;

  if (!stylesheetLink || !shadowCss || !mermaid) {
    throw new Error(
      "The pinned Mermaid and package-derived KaTeX assets must load before mermaid-math.js",
    );
  }

  const stylesheetLoaded = stylesheetLink.sheet
    ? Promise.resolve()
    : new Promise((resolve, reject) => {
        stylesheetLink.addEventListener("load", resolve, { once: true });
        stylesheetLink.addEventListener(
          "error",
          () => reject(new Error("The pinned KaTeX stylesheet failed to load")),
          { once: true },
        );
      });

  // Mermaid asks KaTeX for display-mode output even when a formula is an
  // inline flex item in a node label. KaTeX's default one-em display margin is
  // appropriate between paragraphs, not inside a diagram component. Remove it
  // before Mermaid measures the label, and mirror the same rule in the shadow
  // root so the measured and painted geometries remain identical.
  // Mermaid assigns its prose font to cluster labels after KaTeX has produced
  // the inline formula. Reassert only the outer KaTeX family at that cascade
  // boundary; KaTeX's more specific glyph classes still select their math,
  // size, symbol, and AMS faces. Without this rule, formulas in plate and
  // subgraph captions inherit Mermaid's sans-serif font even though formulas
  // in node and edge labels use KaTeX_Main.
  const diagramMathCss = [
    ".katex-display{margin:0}",
    '.cluster-label .katex{font-family:KaTeX_Main,"Times New Roman",serif!important}',
  ].join("");
  const diagramInteractionCss = [
    "a{cursor:pointer}",
    "a .nodeLabel{text-decoration:underline;text-decoration-thickness:.08em;text-underline-offset:.16em}",
    "a:focus-visible{outline:3px solid var(--artifex-focus,#fbbf24);outline-offset:4px}",
  ].join("");
  const diagramMathStyle = document.createElement("style");
  diagramMathStyle.textContent = diagramMathCss + diagramInteractionCss;
  document.head.append(diagramMathStyle);

  const stylesheetReady = stylesheetLoaded.then(async () => {
    // Font faces belong to the document's font set and are loaded there before
    // measurement. Re-declaring them inside each adopted shadow stylesheet
    // would create a second set of unloaded faces with the same family names.
    // PostCSS generates the exact selector rules without duplicate sources,
    // because browsers cannot inspect linked style sheets under file://.
    const css = `${shadowCss}${diagramMathCss + diagramInteractionCss}`;
    const stylesheet = new CSSStyleSheet();
    await stylesheet.replace(css);
    return stylesheet;
  });

  // KaTeX's supported browser integration lists these families and variants.
  // Load them before Mermaid measures a math-bearing label so the dimensions
  // stored in the SVG match the glyphs that will be painted in the shadow DOM.
  const mathReady = Promise.all([stylesheetLoaded, stylesheetReady]).then(() => {
    const fontFaces = [...document.fonts].filter((fontFace) =>
      fontFace.family.startsWith("KaTeX_"),
    );
    if (fontFaces.length < 20) {
      throw new Error(`Expected 20 KaTeX font faces, found ${fontFaces.length}`);
    }
    return Promise.all(fontFaces.map((fontFace) => fontFace.load()));
  });

  const integration = {
    fontsReady: false,
    katexVersion,
    loadedFontFaceDescriptors: [],
    loadedFontFaces: 0,
    mermaidVersion,
    mathReady,
    stylesheet: undefined,
  };
  mathReady.then((fontFaces) => {
    integration.loadedFontFaces = fontFaces.length;
    integration.fontsReady = fontFaces.every((fontFace) => fontFace.status === "loaded");
    integration.loadedFontFaceDescriptors = fontFaces.map((fontFace) => ({
      family: fontFace.family,
      status: fontFace.status,
      style: fontFace.style,
      weight: fontFace.weight,
    }));
  });
  stylesheetReady.then((stylesheet) => {
    integration.stylesheet = stylesheet;
  });
  window.__artifexMermaidMath = integration;

  // Material initializes Mermaid after this script runs. Mermaid initialization
  // replaces its site configuration, so enforce the supported cross-browser
  // math mode at that boundary rather than mutating rendered SVG afterward.
  const initialize = mermaid.initialize.bind(mermaid);
  mermaid.initialize = (options = {}) =>
    initialize({
      ...options,
      fontFamily: "Roboto, Arial, sans-serif",
      fontSize: 16,
      forceLegacyMathML: true,
      // Book diagrams are trusted, version-controlled sources. Antiscript
      // removes script elements while enabling declarative node links.
      securityLevel: "antiscript",
    });

  const render = mermaid.render.bind(mermaid);
  mermaid.render = async (id, text, container) => {
    if (String(text).includes("$$")) await mathReady;
    return render(id, text, container);
  };

  // Material intentionally inserts each Mermaid SVG into a shadow root. Page
  // CSS cannot cross that boundary, so adopt the exact stylesheet Mermaid used
  // while measuring labels. The render wrapper makes it ready before a math
  // diagram reaches attachShadow; the promise also covers non-math-first pages.
  const nativeAttachShadow = Element.prototype.attachShadow;
  if (!nativeAttachShadow.artifexMermaidInspectable) {
    function attachReadableShadow(init) {
      const isMermaid = this.classList?.contains("mermaid");
      const root = nativeAttachShadow.call(
        this,
        isMermaid ? { ...init, mode: "open" } : init,
      );
      if (isMermaid) {
        const adopt = (stylesheet) => {
          if (!root.adoptedStyleSheets.includes(stylesheet)) {
            root.adoptedStyleSheets = [...root.adoptedStyleSheets, stylesheet];
          }
        };
        if (integration.stylesheet) adopt(integration.stylesheet);
        else stylesheetReady.then(adopt);
      }
      return root;
    }
    attachReadableShadow.artifexMermaidInspectable = true;
    Element.prototype.attachShadow = attachReadableShadow;
  }
})();
