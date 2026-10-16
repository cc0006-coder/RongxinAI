'use strict';

const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');

const {
  resolveRuntimeDownloadSources,
  resolveRuntimeReleaseTag,
  resolveExecutableName,
  extractArchiveToDirectory,
} = require('./download-llamacpp-runtime.cjs');

const ROOT = path.resolve(__dirname, '..');
const BUNDLED_MANIFEST_PATH = path.join(ROOT, 'resources', 'llamacpp-backends', 'manifest.json');
const CPU_EDITION_ROOT = path.join(ROOT, 'vendor', 'llamacpp', 'cpu');

function sha256File(filePath) {
  const hash = crypto.createHash('sha256');
  hash.update(fs.readFileSync(filePath));
  return hash.digest('hex');
}

function resolveRuntimeTargetId(platform = process.platform, arch = process.arch) {
  if (platform === 'win32' && arch === 'x64') return 'win-x64';
  if (platform === 'linux' && arch === 'x64') return 'linux-x64';
  if (platform === 'darwin' && arch === 'arm64') return 'mac-arm64';
  if (platform === 'darwin' && arch === 'x64') return 'mac-x64';
  throw new Error(`Unsupported host for cpu edition sync: ${platform}/${arch}`);
}

function readExpectedArchiveSha256(targetId) {
  if (!fs.existsSync(BUNDLED_MANIFEST_PATH)) return undefined;
  const manifest = JSON.parse(fs.readFileSync(BUNDLED_MANIFEST_PATH, 'utf8'));
  const backendId =
    targetId === 'win-x64'
      ? 'win-x64'
      : targetId === 'linux-x64'
        ? 'linux-x64'
        : targetId === 'mac-arm64'
          ? 'mac-arm64'
          : targetId === 'mac-x64'
            ? 'mac-x64'
            : targetId;
  const entry = manifest.backends?.find(item => item.backend === backendId);
  return entry?.archive?.sha256?.trim().toLowerCase();
}

async function downloadToFile(urls, outputPath) {
  const errors = [];
  for (const url of urls) {
    const response = await fetch(url, {
      headers: { 'User-Agent': 'ZhiYuanAgent/sync-vendor-llamacpp-cpu-edition' },
    });
    if (!response.ok || !response.body) {
      errors.push(`${url} HTTP ${response.status}`);
      continue;
    }
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    const buffer = Buffer.from(await response.arrayBuffer());
    fs.writeFileSync(outputPath, buffer);
    return url;
  }
  throw new Error(`CPU runtime download failed: ${errors.join('; ')}`);
}

function main() {
  const targetId = process.env.LLAMACPP_CPU_TARGET_ID?.trim() || resolveRuntimeTargetId();
  const releaseTag = resolveRuntimeReleaseTag(ROOT);
  const urls = resolveRuntimeDownloadSources(targetId, { rootDir: ROOT });
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'llamacpp-cpu-edition-'));
  const archiveName = path.basename(new URL(urls[0]).pathname);
  const archivePath = path.join(tempDir, archiveName);
  const extractDir = path.join(tempDir, 'extract');

  const run = async () => {
    const sourceUrl = await downloadToFile(urls, archivePath);
    const expectedSha256 = readExpectedArchiveSha256(targetId);
    if (expectedSha256) {
      const actualSha256 = sha256File(archivePath);
      if (actualSha256 !== expectedSha256) {
        throw new Error(
          `CPU runtime archive SHA-256 mismatch for ${targetId}: expected ${expectedSha256}, got ${actualSha256}`,
        );
      }
    }

    fs.mkdirSync(extractDir, { recursive: true });
    await extractArchiveToDirectory(archivePath, extractDir);

    const executableName = resolveExecutableName(targetId);
    const executablePath = findExecutablePath(extractDir, executableName);
    if (!executablePath) {
      throw new Error(`CPU runtime archive does not contain ${executableName}`);
    }

    const binDir = path.join(CPU_EDITION_ROOT, 'bin');
    fs.rmSync(CPU_EDITION_ROOT, { recursive: true, force: true });
    fs.mkdirSync(binDir, { recursive: true });
    copyDirectoryContents(path.dirname(executablePath), binDir);

    const stagedExecutable = path.join(binDir, executableName);
    const manifest = {
      schemaVersion: 1,
      backend: {
        version: releaseTag,
        backend: targetId,
        versionBackend: `${releaseTag}/${targetId}`,
      },
      executable: path.relative(CPU_EDITION_ROOT, stagedExecutable).replace(/\\/g, '/'),
      sourceUrl,
      archiveSha256: sha256File(archivePath),
    };
    manifest.sha256 = sha256File(stagedExecutable);
    fs.writeFileSync(
      path.join(CPU_EDITION_ROOT, 'manifest.json'),
      `${JSON.stringify(manifest, null, 2)}\n`,
    );
    console.log(
      `[sync-vendor-llamacpp-cpu-edition] Staged ${targetId} cpu edition at ${CPU_EDITION_ROOT} (${manifest.sha256.slice(0, 12)}…).`,
    );
  };

  run()
    .catch(error => {
      console.error('[sync-vendor-llamacpp-cpu-edition]', error instanceof Error ? error.message : error);
      process.exitCode = 1;
    })
    .finally(() => {
      fs.rmSync(tempDir, { recursive: true, force: true });
    });
}

function findExecutablePath(rootDir, executableName) {
  const queue = [rootDir];
  while (queue.length > 0) {
    const current = queue.shift();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) queue.push(fullPath);
      else if (entry.name === executableName) return fullPath;
    }
  }
  return null;
}

function copyDirectoryContents(sourceDir, targetDir) {
  for (const entry of fs.readdirSync(sourceDir, { withFileTypes: true })) {
    const sourcePath = path.join(sourceDir, entry.name);
    const targetPath = path.join(targetDir, entry.name);
    if (entry.isDirectory()) {
      fs.mkdirSync(targetPath, { recursive: true });
      copyDirectoryContents(sourcePath, targetPath);
      continue;
    }
    fs.copyFileSync(sourcePath, targetPath);
  }
}

if (require.main === module) {
  main();
}

module.exports = { CPU_EDITION_ROOT, resolveRuntimeTargetId };
