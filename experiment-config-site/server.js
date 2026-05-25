const http = require("http");
const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const PORT = Number(process.env.PORT || 4173);
const HOST = process.env.HOST || "127.0.0.1";
const SITE_ROOT = __dirname;
const REPO_ROOT = path.resolve(SITE_ROOT, "..");
const PUBLIC_ROOT = path.join(SITE_ROOT, "public");
const READER = path.join(SITE_ROOT, "src", "read_experiments.py");
const CONTACT_VIZ_ROOT = path.join(REPO_ROOT, "tool_contact_viz");

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
};

let cache = null;

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload, null, 2);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
  });
  res.end(body);
}

function sendText(res, status, body) {
  res.writeHead(status, { "content-type": "text/plain; charset=utf-8" });
  res.end(body);
}

function getConfigMtime() {
  const configDir = path.join(REPO_ROOT, "configs", "experiments");
  try {
    return fs
      .readdirSync(configDir)
      .filter((name) => name.endsWith(".py"))
      .map((name) => fs.statSync(path.join(configDir, name)).mtimeMs)
      .reduce((max, value) => Math.max(max, value), 0);
  } catch (error) {
    return 0;
  }
}

function loadExperiments({ force = false } = {}) {
  const mtime = getConfigMtime();
  if (!force && cache && cache.mtime === mtime) {
    return cache.payload;
  }

  const command = process.env.PYTHON || "python";
  const result = spawnSync(command, [READER, REPO_ROOT], {
    cwd: REPO_ROOT,
    encoding: "utf8",
    maxBuffer: 20 * 1024 * 1024,
  });

  if (result.error) {
    throw new Error(`Failed to run Python reader: ${result.error.message}`);
  }
  if (result.status !== 0) {
    throw new Error(result.stderr || `Python reader exited with ${result.status}`);
  }

  const payload = JSON.parse(result.stdout);
  cache = { mtime, payload, loadedAt: Date.now() };
  return payload;
}

function getExperimentIdFromPath(pathname) {
  const prefix = "/api/experiments/";
  if (!pathname.startsWith(prefix)) {
    return null;
  }
  return decodeURIComponent(pathname.slice(prefix.length));
}

function displayNameFromSlug(slug) {
  return slug
    .split("_")
    .filter(Boolean)
    .map((part) => (part.length === 1 ? part.toUpperCase() : part[0].toUpperCase() + part.slice(1)))
    .join(" ");
}

function loadContactVizGroups() {
  const files = fs
    .readdirSync(CONTACT_VIZ_ROOT)
    .filter((name) => name.endsWith(".png"))
    .sort();
  const groups = new Map();

  for (const file of files) {
    const match = file.match(/^(\d{3})_(.+)_end_effector_(pre|post)\.png$/);
    if (!match) {
      continue;
    }
    const [, indexText, slug, kind] = match;
    const id = `${indexText}_${slug}`;
    const group =
      groups.get(id) ||
      {
        id,
        index: Number(indexText),
        slug,
        label: `${indexText} ${displayNameFromSlug(slug)}`,
      };
    group[kind] = {
      file,
      url: `/tool-contact-viz/${encodeURIComponent(file)}`,
    };
    groups.set(id, group);
  }

  return Array.from(groups.values())
    .filter((group) => group.pre && group.post)
    .sort((a, b) => a.index - b.index);
}

function serveContactVizImage(res, pathname) {
  const prefix = "/tool-contact-viz/";
  const filename = decodeURIComponent(pathname.slice(prefix.length));
  const absolute = path.join(CONTACT_VIZ_ROOT, filename);

  if (!absolute.startsWith(CONTACT_VIZ_ROOT) || path.basename(filename) !== filename) {
    sendText(res, 403, "Forbidden");
    return;
  }

  fs.readFile(absolute, (error, content) => {
    if (error) {
      sendText(res, 404, "Not found");
      return;
    }
    res.writeHead(200, {
      "content-type": MIME_TYPES[path.extname(absolute)] || "application/octet-stream",
      "cache-control": "public, max-age=3600",
    });
    res.end(content);
  });
}

function serveStatic(res, pathname) {
  const normalized = pathname === "/" ? "/index.html" : pathname;
  const requested = path.normalize(decodeURIComponent(normalized)).replace(/^(\.\.[/\\])+/, "");
  const absolute = path.join(PUBLIC_ROOT, requested);

  if (!absolute.startsWith(PUBLIC_ROOT)) {
    sendText(res, 403, "Forbidden");
    return;
  }

  fs.readFile(absolute, (error, content) => {
    if (error) {
      fs.readFile(path.join(PUBLIC_ROOT, "index.html"), (fallbackError, fallbackContent) => {
        if (fallbackError) {
          sendText(res, 404, "Not found");
          return;
        }
        res.writeHead(200, { "content-type": MIME_TYPES[".html"] });
        res.end(fallbackContent);
      });
      return;
    }

    const ext = path.extname(absolute);
    res.writeHead(200, { "content-type": MIME_TYPES[ext] || "application/octet-stream" });
    res.end(content);
  });
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || `${HOST}:${PORT}`}`);
  const pathname = url.pathname;

  try {
    if (pathname === "/api/health") {
      sendJson(res, 200, { ok: true });
      return;
    }

    if (pathname === "/api/experiments") {
      const payload = loadExperiments({ force: url.searchParams.get("refresh") === "1" });
      sendJson(res, 200, {
        generatedAt: payload.generatedAt,
        count: payload.experiments.length,
        experiments: payload.experiments.map(
          ({ fullConfig, sourceText, assignments, artifacts, ...summary }) => summary,
        ),
        artifactRoot: payload.artifactRoot,
        artifactCount: payload.artifactCount,
        errors: payload.errors,
      });
      return;
    }

    if (pathname === "/api/contact-viz") {
      const groups = loadContactVizGroups();
      sendJson(res, 200, {
        count: groups.length,
        groups,
      });
      return;
    }

    if (pathname.startsWith("/tool-contact-viz/")) {
      serveContactVizImage(res, pathname);
      return;
    }

    const experimentId = getExperimentIdFromPath(pathname);
    if (experimentId) {
      const payload = loadExperiments();
      const experiment = payload.experiments.find((item) => item.id === experimentId);
      if (!experiment) {
        sendJson(res, 404, { error: "Experiment not found" });
        return;
      }
      sendJson(res, 200, experiment);
      return;
    }

    serveStatic(res, pathname);
  } catch (error) {
    sendJson(res, 500, { error: error.message });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`Experiment config site running at http://${HOST}:${PORT}`);
});
