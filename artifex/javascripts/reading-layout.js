(() => {
  const storageKeys = {
    navigation: "artifex.docs.navigation-collapsed",
    contents: "artifex.docs.contents-collapsed",
  };

  const readPreference = (key) => {
    try {
      return window.localStorage.getItem(key) === "true";
    } catch {
      return false;
    }
  };

  const writePreference = (key, value) => {
    try {
      window.localStorage.setItem(key, String(value));
    } catch {
      // Private browsing or a locked-down embed may deny storage. The current
      // page still receives the requested layout state.
    }
  };

  const state = {
    navigation: readPreference(storageKeys.navigation),
    contents: readPreference(storageKeys.contents),
  };
  const minimumMermaidNodeFontSize = 12;
  const positionedMermaidSvgs = new WeakSet();
  const mermaidDiagramStates = new WeakMap();
  const mermaidLinkLists = new WeakMap();
  const mermaidZoomFactor = 1.25;
  let mermaidDiagramNumber = 0;
  // Material 3's container/on-container role pairing is applied to diagram
  // components, while shape and text keep meaning independent of color. Each
  // pair is checked at runtime against WCAG text and graphical-object ratios.
  const mermaidComponentPalette = {
    light: {
      observed: { surface: "#d8e1f0", content: "#111827", boundary: "#40536d" },
      rect: { surface: "#eef2ff", content: "#1f2328", boundary: "#5261a8" },
      polygon: { surface: "#fff5d6", content: "#1f2328", boundary: "#806000" },
      circle: { surface: "#e8f7f0", content: "#1f2328", boundary: "#287057" },
      ellipse: { surface: "#e8f7f0", content: "#1f2328", boundary: "#287057" },
      path: { surface: "#f5efff", content: "#1f2328", boundary: "#715197" },
      connector: {
        stroke: "#596579",
        labelSurface: "#ffffff",
        labelContent: "#1f2328",
        clusterSurface: "#f6f7fb",
        clusterBoundary: "#596579",
      },
    },
    dark: {
      observed: { surface: "#526176", content: "#ffffff", boundary: "#c5d2e3" },
      rect: { surface: "#30364a", content: "#f2f4f8", boundary: "#aebaff" },
      polygon: { surface: "#463b25", content: "#f2f4f8", boundary: "#f2cf72" },
      circle: { surface: "#263e37", content: "#f2f4f8", boundary: "#8bd8b9" },
      ellipse: { surface: "#263e37", content: "#f2f4f8", boundary: "#8bd8b9" },
      path: { surface: "#3b3049", content: "#f2f4f8", boundary: "#c8aae8" },
      connector: {
        stroke: "#b8c1d1",
        labelSurface: "#252a34",
        labelContent: "#f2f4f8",
        clusterSurface: "#2d333f",
        clusterBoundary: "#8490a3",
      },
    },
  };
  let mermaidObserver;
  let mermaidPaletteObserver;
  let mermaidFrame;
  let readingProgressFrame;
  let tableFrame;

  const applyMermaidPalette = (host, root, dark) => {
    const palette = mermaidComponentPalette[dark ? "dark" : "light"];
    const connector = palette.connector;

    for (const node of root.querySelectorAll(".node")) {
      const shape = node.querySelector("rect, polygon, circle, ellipse, path");
      if (!shape) continue;
      // Mermaid retains `class <node> observed` on the rendered node group.
      // This semantic role takes precedence over geometry: an observed circle
      // must remain visibly observed in every plate diagram, while an unshaded
      // circle continues to denote a latent random variable.
      const role = node.classList.contains("observed")
        ? "observed"
        : shape.tagName.toLowerCase();
      const colors = palette[role] || palette.path;
      shape.style.setProperty("fill", colors.surface, "important");
      shape.style.setProperty("stroke", colors.boundary, "important");
      shape.style.setProperty("stroke-width", "1.5px", "important");
      for (const label of node.querySelectorAll(".nodeLabel, .nodeLabel *, .label, .label *")) {
        label.style.setProperty("color", colors.content, "important");
      }
    }
    for (const label of root.querySelectorAll(".edgeLabel, .edgeLabel *")) {
      label.style.setProperty("color", connector.labelContent, "important");
      label.style.setProperty("background-color", connector.labelSurface, "important");
    }
    for (const edge of root.querySelectorAll(
      ".flowchart-link, .edgePath path.path, [marker-end]",
    )) {
      edge.style.setProperty("stroke", connector.stroke, "important");
      edge.style.setProperty("stroke-width", "2px", "important");
    }
    for (const marker of root.querySelectorAll("marker path")) {
      marker.style.setProperty("fill", connector.stroke, "important");
      marker.style.setProperty("stroke", connector.stroke, "important");
    }
    for (const cluster of root.querySelectorAll(".cluster rect")) {
      cluster.style.setProperty("fill", connector.clusterSurface, "important");
      cluster.style.setProperty("stroke", connector.clusterBoundary, "important");
    }
    for (const label of root.querySelectorAll(".cluster-label, .cluster-label *")) {
      label.style.setProperty("color", connector.labelContent, "important");
    }
    host.dataset.artifexMermaidPalette = dark ? "dark" : "light";
  };

  const accessibleText = (element) => {
    const clone = element?.cloneNode(true);
    if (!clone) return "";
    for (const decoration of clone.querySelectorAll(".headerlink, [aria-hidden='true']")) {
      decoration.remove();
    }
    return clone.textContent?.replace(/\s+/g, " ").trim() || "";
  };

  const nearestDiagramHeading = (host) => {
    const article = host.closest("article");
    if (!article) return "Generative-model diagram";
    const headings = [...article.querySelectorAll("h1, h2, h3, h4, h5, h6")];
    const heading = headings
      .filter((heading) => heading.compareDocumentPosition(host) & Node.DOCUMENT_POSITION_FOLLOWING)
      .at(-1);
    return accessibleText(heading) || "Generative-model diagram";
  };

  const enhanceMermaidLinks = (host, root, svg) => {
    // The renderer supplies a graphics-document role. Give every SVG a short
    // name even while source-authored long descriptions are completed.
    if (!svg.querySelector("title")) {
      const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
      title.id = `${host.id}-title`;
      title.textContent = nearestDiagramHeading(host);
      svg.prepend(title);
      svg.setAttribute("aria-labelledby", title.id);
    }

    const destinations = new Map();
    for (const link of root.querySelectorAll("svg a")) {
      const href = link.getAttribute("href") || link.getAttribute("xlink:href");
      if (!href) continue;
      // SVG 2 uses `href`; retaining Mermaid's `xlink:href` keeps compatibility
      // while making the destination recognizable to current crawlers.
      if (!link.hasAttribute("href")) link.setAttribute("href", href);
      const label = link.querySelector("[title]")?.getAttribute("title")
        || accessibleText(link)
        || "Open diagram destination";
      destinations.set(`${href}\u0000${label}`, { href, label });
    }

    let details = mermaidLinkLists.get(host);
    if (!destinations.size) {
      details?.remove();
      mermaidLinkLists.delete(host);
      return;
    }
    if (!details) {
      details = document.createElement("details");
      details.className = "artifex-diagram-links";
      details.dataset.artifexDiagramLinks = "true";
      details.append(document.createElement("summary"), document.createElement("ul"));
      mermaidLinkLists.set(host, details);
    }

    const signature = JSON.stringify([...destinations.values()]);
    if (details.dataset.artifexDiagramLinkSignature !== signature) {
      details.dataset.artifexDiagramLinkSignature = signature;
      details.querySelector("summary").textContent =
        `Diagram destinations (${destinations.size})`;
      const list = details.querySelector("ul");
      list.replaceChildren();
      for (const { href, label } of destinations.values()) {
        const item = document.createElement("li");
        const link = document.createElement("a");
        link.href = href;
        link.textContent = label;
        item.append(link);
        list.append(item);
      }
    }
    const diagramBlock = host.closest(".artifex-diagram-shell") || host;
    if (diagramBlock.nextElementSibling !== details) diagramBlock.after(details);
  };

  const makeTablesKeyboardScrollable = () => {
    tableFrame = undefined;
    for (const wrapper of document.querySelectorAll("article .md-typeset__table")) {
      const overflow = getComputedStyle(wrapper).overflowX;
      const scrollable = ["auto", "scroll"].includes(overflow)
        && wrapper.scrollWidth > wrapper.clientWidth + 1;
      if (scrollable) {
        if (!wrapper.hasAttribute("tabindex")) {
          wrapper.tabIndex = 0;
          wrapper.dataset.artifexKeyboardScroll = "true";
          wrapper.setAttribute("role", "group");
          wrapper.setAttribute(
            "aria-label",
            `Scrollable table: ${nearestDiagramHeading(wrapper)}`,
          );
        }
      } else if (wrapper.dataset.artifexKeyboardScroll === "true") {
        wrapper.removeAttribute("tabindex");
        wrapper.removeAttribute("role");
        wrapper.removeAttribute("aria-label");
        delete wrapper.dataset.artifexKeyboardScroll;
      }
    }
  };

  const labelCodeBlockNavigation = () => {
    for (const [index, navigation] of document
      .querySelectorAll("article .md-code__nav")
      .entries()) {
      navigation.setAttribute("aria-label", `Code block ${index + 1} actions`);
    }
  };

  const scheduleTableReadability = () => {
    if (tableFrame !== undefined) return;
    tableFrame = window.requestAnimationFrame(makeTablesKeyboardScrollable);
  };

  const clamp = (value, lower, upper) => Math.min(upper, Math.max(lower, value));

  const setDiagramWidth = (diagram, width, mode, preserveCenter = true) => {
    const { host, svg } = diagram;
    const oldWidth = svg.getBoundingClientRect().width || diagram.currentWidth || width;
    const oldHeight = svg.getBoundingClientRect().height ||
      oldWidth * diagram.naturalHeight / diagram.naturalWidth;
    const oldCenter = host.scrollLeft + host.clientWidth / 2;
    const oldMiddle = host.scrollTop + host.clientHeight / 2;
    const centerFraction = oldWidth ? oldCenter / oldWidth : 0.5;
    const middleFraction = oldHeight ? oldMiddle / oldHeight : 0.5;
    const nextWidth = Math.round(clamp(width, diagram.fitWidth, diagram.maximumWidth));
    const nextHeight = nextWidth * diagram.naturalHeight / diagram.naturalWidth;

    diagram.currentWidth = nextWidth;
    diagram.mode = mode;
    svg.style.setProperty("width", `${nextWidth}px`, "important");
    svg.style.setProperty("max-width", "none", "important");
    svg.style.setProperty("height", "auto", "important");
    const scrollable = nextWidth > host.clientWidth + 1 ||
      nextHeight > diagram.viewportHeight + 1;
    host.classList.toggle("artifex-mermaid-scrollable", scrollable);

    if (diagram.output) {
      const percentage = Math.round(100 * nextWidth / diagram.naturalWidth);
      const label = `${percentage}%`;
      if (diagram.output.value !== label) diagram.output.value = label;
      if (diagram.output.textContent !== label) diagram.output.textContent = label;
    }
    if (diagram.buttons) {
      diagram.buttons.out.disabled = nextWidth <= diagram.fitWidth + 1;
      diagram.buttons.in.disabled = nextWidth >= diagram.maximumWidth - 1;
    }

    if (preserveCenter && scrollable) {
      window.requestAnimationFrame(() => {
        if (!host.isConnected || !svg.isConnected) return;
        host.scrollLeft = Math.max(0, centerFraction * nextWidth - host.clientWidth / 2);
        host.scrollTop = Math.max(0, middleFraction * nextHeight - host.clientHeight / 2);
      });
    } else if (!scrollable) {
      host.scrollLeft = 0;
      host.scrollTop = 0;
    }
  };

  const makeDiagramButton = (action, label, text) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "artifex-diagram-button";
    button.dataset.artifexDiagramAction = action;
    button.setAttribute("aria-label", label);
    button.title = label;
    button.textContent = text;
    return button;
  };

  const ensureDiagramNavigator = (diagram) => {
    const { host } = diagram;
    if (diagram.shell?.isConnected) return;

    const instructionsId = `${host.id}-instructions`;
    const shell = document.createElement("div");
    shell.className = "artifex-diagram-shell";
    shell.dataset.artifexDiagramNavigator = "true";
    host.before(shell);
    shell.append(host);

    const toolbar = document.createElement("div");
    toolbar.className = "artifex-diagram-toolbar";
    toolbar.setAttribute("role", "group");
    toolbar.setAttribute("aria-label", `Diagram view controls for ${nearestDiagramHeading(host)}`);

    const fit = makeDiagramButton("fit", "Fit the whole diagram", "Fit");
    const out = makeDiagramButton("out", "Zoom out", "−");
    const reset = makeDiagramButton("reset", "Restore the readable diagram size", "Readable");
    const into = makeDiagramButton("in", "Zoom in", "+");
    const output = document.createElement("output");
    output.className = "artifex-diagram-zoom";
    output.setAttribute("aria-live", "polite");
    output.setAttribute("aria-label", "Current diagram zoom");
    toolbar.append(fit, out, output, into, reset);

    const instructions = document.createElement("p");
    instructions.id = instructionsId;
    instructions.className = "artifex-visually-hidden";
    instructions.textContent =
      "Use Fit, Zoom out, Zoom in, or Readable. Drag the diagram with a mouse or pen, scroll it with one finger, or focus it and use the arrow keys to pan. Press F to fit, plus or minus to zoom, and zero to restore the readable size.";

    shell.prepend(toolbar);
    shell.append(instructions);
    host.tabIndex = 0;
    host.setAttribute("role", "region");
    host.setAttribute("aria-label", `${nearestDiagramHeading(host)} interactive diagram`);
    host.setAttribute("aria-describedby", instructionsId);

    diagram.shell = shell;
    diagram.toolbar = toolbar;
    diagram.output = output;
    diagram.buttons = { fit, out, in: into, reset };

    const changeView = (action) => {
      if (action === "fit") {
        setDiagramWidth(diagram, diagram.fitWidth, "fit");
      } else if (action === "reset") {
        setDiagramWidth(diagram, diagram.readableWidth, "readable");
      } else if (action === "in") {
        setDiagramWidth(diagram, diagram.currentWidth * mermaidZoomFactor, "custom");
      } else if (action === "out") {
        setDiagramWidth(diagram, diagram.currentWidth / mermaidZoomFactor, "custom");
      }
    };

    toolbar.addEventListener("click", (event) => {
      const button = event.target.closest("[data-artifex-diagram-action]");
      if (button && !button.disabled) changeView(button.dataset.artifexDiagramAction);
    });
    host.addEventListener("keydown", (event) => {
      if (event.target !== host || event.ctrlKey || event.metaKey || event.altKey) return;
      const action = {
        f: "fit",
        F: "fit",
        "0": "reset",
        "+": "in",
        "=": "in",
        "-": "out",
        _: "out",
      }[event.key];
      if (action) {
        event.preventDefault();
        changeView(action);
        return;
      }
      const horizontalPan = Math.max(48, Math.round(host.clientWidth * 0.18));
      const verticalPan = Math.max(48, Math.round(host.clientHeight * 0.18));
      if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
        event.preventDefault();
        host.scrollBy({
          left: event.key === "ArrowLeft"
            ? -horizontalPan
            : event.key === "ArrowRight"
              ? horizontalPan
              : 0,
          top: event.key === "ArrowUp"
            ? -verticalPan
            : event.key === "ArrowDown"
              ? verticalPan
              : 0,
        });
      } else if (event.key === "Home" || event.key === "End") {
        event.preventDefault();
        host.scrollLeft = event.key === "Home" ? 0 : host.scrollWidth;
        host.scrollTop = event.key === "Home" ? 0 : host.scrollHeight;
      }
    });

    let pointerPan = null;
    host.addEventListener("pointerdown", (event) => {
      if (
        event.pointerType === "touch" ||
        event.button !== 0 ||
        event.composedPath().some((element) => element instanceof Element && element.closest?.("a"))
      ) return;
      pointerPan = {
        id: event.pointerId,
        left: host.scrollLeft,
        top: host.scrollTop,
        x: event.clientX,
        y: event.clientY,
      };
      host.setPointerCapture(event.pointerId);
      host.classList.add("artifex-diagram-panning");
      host.focus({ preventScroll: true });
      event.preventDefault();
    });
    host.addEventListener("pointermove", (event) => {
      if (!pointerPan || event.pointerId !== pointerPan.id) return;
      host.scrollLeft = pointerPan.left - (event.clientX - pointerPan.x);
      host.scrollTop = pointerPan.top - (event.clientY - pointerPan.y);
      event.preventDefault();
    });
    const endPointerPan = (event) => {
      if (!pointerPan || event.pointerId !== pointerPan.id) return;
      if (host.hasPointerCapture(event.pointerId)) host.releasePointerCapture(event.pointerId);
      pointerPan = null;
      host.classList.remove("artifex-diagram-panning");
    };
    host.addEventListener("pointerup", endPointerPan);
    host.addEventListener("pointercancel", endPointerPan);
  };

  const makeMermaidsReadable = () => {
    mermaidFrame = undefined;
    for (const host of document.querySelectorAll("article .mermaid")) {
      const root = host.shadowRoot || host;
      const svg = root.querySelector("svg");
      const viewBox = svg?.viewBox?.baseVal;
      const labels = svg ? [...svg.querySelectorAll(".node .nodeLabel, .node .label")] : [];
      if (!svg || !viewBox?.width || !viewBox.height || !labels.length || !host.clientWidth) continue;
      if (!host.id) {
        mermaidDiagramNumber += 1;
        host.id = `artifex-mermaid-${mermaidDiagramNumber}`;
      }
      enhanceMermaidLinks(host, root, svg);

      const fontSizes = labels
        .map((label) => Number.parseFloat(getComputedStyle(label).fontSize))
        .filter((size) => Number.isFinite(size) && size > 0);
      if (!fontSizes.length) continue;

      const smallestBaseFont = Math.min(...fontSizes);
      const rootFontSize = Number.parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
      const maximumHostHeight = Math.max(
        rootFontSize * 10,
        Math.min(window.innerHeight * 0.72, rootFontSize * 52),
      );
      let diagram = mermaidDiagramStates.get(host);
      const isNewDiagram = !diagram || diagram.svg !== svg;
      if (isNewDiagram) {
        diagram = {
          host,
          svg,
          mode: "readable",
          currentWidth: viewBox.width,
        };
        mermaidDiagramStates.set(host, diagram);
      }
      const needsNavigator = viewBox.width > host.clientWidth + 1 ||
        viewBox.height > maximumHostHeight + 1;
      if (needsNavigator) ensureDiagramNavigator(diagram);

      // `max-height` constrains the host's border box, whereas the SVG lives
      // in its content box. Measure the global shell after it exists instead
      // of duplicating its padding in JavaScript. The small tolerance also
      // prevents a fractional-pixel scrollbar from surviving the Fit action.
      const hostStyle = getComputedStyle(host);
      const styledMaximumHeight = Number.parseFloat(hostStyle.maxHeight);
      const hostHeightLimit = Number.isFinite(styledMaximumHeight)
        ? styledMaximumHeight
        : maximumHostHeight;
      const verticalInsets = [
        hostStyle.paddingTop,
        hostStyle.paddingBottom,
        hostStyle.borderTopWidth,
        hostStyle.borderBottomWidth,
      ].reduce((total, value) => total + (Number.parseFloat(value) || 0), 0);
      const viewportHeight = Math.max(
        rootFontSize * 8,
        hostHeightLimit - verticalInsets - 4,
      );
      const fitWidth = Math.min(
        viewBox.width,
        host.clientWidth,
        viewBox.width * viewportHeight / viewBox.height,
      );
      const readableWidth = Math.ceil(
        Math.min(
          viewBox.width,
          Math.max(
            host.clientWidth,
            viewBox.width * minimumMermaidNodeFontSize / smallestBaseFont,
          ),
        ),
      );
      if (isNewDiagram) diagram.currentWidth = readableWidth;
      diagram.naturalWidth = viewBox.width;
      diagram.naturalHeight = viewBox.height;
      diagram.viewportHeight = viewportHeight;
      diagram.fitWidth = fitWidth;
      diagram.readableWidth = readableWidth;
      diagram.maximumWidth = Math.max(viewBox.width * 2, readableWidth);

      const targetWidth = diagram.mode === "fit"
        ? fitWidth
        : diagram.mode === "readable"
          ? readableWidth
          : clamp(diagram.currentWidth, fitWidth, diagram.maximumWidth);
      setDiagramWidth(diagram, targetWidth, diagram.mode, false);
      const scrollable = targetWidth > host.clientWidth + 1;
      applyMermaidPalette(
        host,
        root,
        document.body.dataset.mdColorScheme === "slate",
      );

      if (scrollable && !positionedMermaidSvgs.has(svg)) {
        positionedMermaidSvgs.add(svg);
        window.requestAnimationFrame(() => {
          // The navigator wrapper changes the host's bounded viewport. Wait a
          // second frame so centering uses its final two-dimensional geometry.
          window.requestAnimationFrame(() => {
            const entryNode = svg.querySelector(".node.entry, .node");
            if (!entryNode?.isConnected || !host.isConnected) return;
            const hostBox = host.getBoundingClientRect();
            const nodeBox = entryNode.getBoundingClientRect();
            const nodeCenter = nodeBox.left - hostBox.left + host.scrollLeft + nodeBox.width / 2;
            const nodeMiddle = nodeBox.top - hostBox.top + host.scrollTop + nodeBox.height / 2;
            host.scrollLeft = Math.max(0, nodeCenter - host.clientWidth / 2);
            host.scrollTop = Math.max(0, nodeMiddle - host.clientHeight / 2);
          });
        });
      }
    }
  };

  const scheduleMermaidReadability = () => {
    if (mermaidFrame !== undefined) return;
    mermaidFrame = window.requestAnimationFrame(makeMermaidsReadable);
  };

  const scheduleMermaidReadabilityChecks = () => {
    scheduleMermaidReadability();
    scheduleTableReadability();
    window.setTimeout(scheduleMermaidReadability, 100);
    window.setTimeout(scheduleMermaidReadability, 500);
    window.setTimeout(scheduleTableReadability, 100);
    window.setTimeout(scheduleTableReadability, 500);
  };

  const watchMermaids = (content) => {
    mermaidObserver?.disconnect();
    mermaidObserver = new MutationObserver(scheduleMermaidReadabilityChecks);
    mermaidObserver.observe(content, { childList: true, subtree: true });
    scheduleMermaidReadabilityChecks();
    window.setTimeout(scheduleMermaidReadability, 250);
    window.setTimeout(scheduleMermaidReadability, 1000);

    mermaidPaletteObserver?.disconnect();
    mermaidPaletteObserver = new MutationObserver(scheduleMermaidReadabilityChecks);
    mermaidPaletteObserver.observe(document.body, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"],
    });
  };

  const applyState = () => {
    document.body.classList.toggle("artifex-navigation-collapsed", state.navigation);
    document.body.classList.toggle("artifex-contents-collapsed", state.contents);

    for (const button of document.querySelectorAll("[data-artifex-sidebar]")) {
      const side = button.dataset.artifexSidebar;
      const collapsed = state[side];
      const readableName = side === "navigation" ? "site navigation" : "page contents";
      button.setAttribute("aria-expanded", String(!collapsed));
      button.setAttribute("aria-label", `${collapsed ? "Show" : "Hide"} ${readableName}`);
      button.title = `${collapsed ? "Show" : "Hide"} ${readableName}`;
      button.classList.toggle("is-collapsed", collapsed);
    }
  };

  const updateReadingProgress = () => {
    readingProgressFrame = undefined;
    const article = document.querySelector(".md-content article");
    const progress = document.querySelector("[data-artifex-reading-progress]");
    if (!article || !progress) return;
    const articleTop = article.getBoundingClientRect().top + window.scrollY;
    const finalScroll = articleTop + article.offsetHeight - window.innerHeight;
    const fraction = finalScroll <= articleTop
      ? 1
      : (window.scrollY - articleTop) / (finalScroll - articleTop);
    const percent = Math.round(100 * Math.min(1, Math.max(0, fraction)));
    progress.value = percent;
    progress.setAttribute("aria-valuetext", `${percent}% of chapter read`);
    progress.title = `${percent}% of chapter read`;
  };

  const scheduleReadingProgress = () => {
    if (readingProgressFrame !== undefined) return;
    readingProgressFrame = window.requestAnimationFrame(updateReadingProgress);
  };

  const mountReadingProgress = (content) => {
    if (content.querySelector(".artifex-reading-progress")) {
      scheduleReadingProgress();
      return;
    }
    const container = document.createElement("div");
    container.className = "artifex-reading-progress";
    const progress = document.createElement("progress");
    progress.dataset.artifexReadingProgress = "true";
    progress.max = 100;
    progress.value = 0;
    progress.setAttribute("aria-label", "Chapter reading progress");
    progress.setAttribute("aria-valuetext", "0% of chapter read");
    container.append(progress);
    content.prepend(container);
    scheduleReadingProgress();
  };

  const captureReadingPosition = () => {
    const midpoint = window.innerHeight / 2;
    const blocks = [...document.querySelectorAll(
      "article h1, article h2, article h3, article h4, article p, article li, article tr, article pre, article blockquote, article details, article .admonition, article .arithmatex, article .mermaid",
    )]
      .map((element) => ({ element, rect: element.getBoundingClientRect() }))
      .filter(({ rect }) => rect.height > 0 && rect.bottom > 0 && rect.top < window.innerHeight);
    const intersecting = blocks
      .filter(({ rect }) => rect.top <= midpoint && rect.bottom >= midpoint)
      .sort((left, right) => left.rect.height - right.rect.height);
    const nearest = [...blocks].sort(
      (left, right) => Math.abs(left.rect.top - midpoint) - Math.abs(right.rect.top - midpoint),
    );
    const block = (intersecting[0] || nearest[0])?.element;
    return block ? { element: block, top: block.getBoundingClientRect().top } : null;
  };

  const restoreReadingPosition = (position) => {
    if (!position) return;
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        if (!position.element.isConnected) return;
        const delta = position.element.getBoundingClientRect().top - position.top;
        if (Math.abs(delta) > 0.5) window.scrollBy(0, delta);
      });
    });
  };

  const makeButton = (side, label, controls) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "artifex-layout-button";
    button.dataset.artifexSidebar = side;
    button.setAttribute("aria-controls", controls);
    button.innerHTML = `
      <span class="artifex-layout-button__icon" aria-hidden="true">${side === "navigation" ? "☰" : "≡"}</span>
      <span>${label}</span>
    `;
    button.addEventListener("click", () => {
      const readingPosition = captureReadingPosition();
      state[side] = !state[side];
      writePreference(storageKeys[side], state[side]);
      applyState();
      restoreReadingPosition(readingPosition);
    });
    return button;
  };

  const mount = () => {
    const content = document.querySelector(".md-content");
    if (!content) {
      applyState();
      return;
    }
    mountReadingProgress(content);
    if (content.querySelector(".artifex-layout-toolbar")) {
      applyState();
      return;
    }

    const primary = document.querySelector(".md-sidebar--primary");
    const secondary = document.querySelector(".md-sidebar--secondary");
    if (primary) primary.id ||= "artifex-navigation-sidebar";
    if (secondary) secondary.id ||= "artifex-contents-sidebar";

    const toolbar = document.createElement("div");
    toolbar.className = "artifex-layout-toolbar";
    toolbar.setAttribute("role", "group");
    toolbar.setAttribute("aria-label", "Reading layout");

    if (primary) {
      toolbar.append(makeButton("navigation", "Navigation", primary.id));
    }
    if (secondary) {
      toolbar.append(makeButton("contents", "Contents", secondary.id));
    }
    content.prepend(toolbar);
    applyState();
    labelCodeBlockNavigation();
    makeTablesKeyboardScrollable();
    watchMermaids(content);
  };

  const initialize = () => {
    mount();
    const content = document.querySelector(".md-content");
    if (content) watchMermaids(content);
  };

  if (typeof document$ !== "undefined") {
    document$.subscribe(initialize);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
  window.addEventListener("resize", scheduleTableReadability, { passive: true });
  window.addEventListener("scroll", scheduleReadingProgress, { passive: true });
  window.addEventListener("resize", scheduleReadingProgress, { passive: true });

  window.addEventListener("storage", (event) => {
    for (const [side, key] of Object.entries(storageKeys)) {
      if (event.key === key) state[side] = event.newValue === "true";
    }
    applyState();
  });
  window.addEventListener("resize", scheduleMermaidReadability, { passive: true });
})();
