'use strict';

const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const BUNDLED_DIRECTORY_NAME = 'bundled';
const BUILD_INFO_FILE = 'runtime-build-info.json';
const CHECKSUMS_FILE = path.join('resources', 'llamacpp-backends', 'bundle-checksums.json');

const BUNDLE_TARGETS = {
  'win32/x64': [
    {
      backend: 'win-x64',
      accelerator: 'cpu',
      assetTemplate: 'llama-{tag}-bin-win-cpu-x64.zip',
    },
    {
      backend: 'win-x64-vulkan',
      accelerator: 'vulkan',
      assetTemplate: 'llama-{tag}-bin-win-vulkan-x64.zip',
    },
  ],
  'linux/x64': [
    {
      backend: 'linux-x64',
      accelerator: 'cpu',
      assetTemplate: 'llama-{tag}-bin-ubuntu-x64.tar.gz',
    },
    {
      backend: 'linux-x64-vulkan',
      accelerator: 'vulkan',
      assetTemplate: 'llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz',
    },
  ],
  'linux/arm64': [
    {
      backend: 'linux-arm64',
      accelerator: 'cpu',
      assetTemplate: 'llama-{tag}-bin-ubuntu-arm64.tar.gz',
    },
    {
      backend: 'linux-arm64-vulkan',
      accelerator: 'vulkan',
      assetTemplate: 'llama-{tag}-bin-ubuntu-vulkan-arm64.tar.gz',
    },
  ],
  'darwin/arm64': [
    {
      backend: 'mac-arm64',
      accelerator: 'metal',
      assetTemplate: 'llama-{tag}-bin-macos-arm64.tar.gz',
    },
  ],
  'darwin/x64': [
    {
      backend: 'mac-x64',
      accelerator: 'metal',
      assetTemplate: 'llama-{tag}-bin-macos-x64.tar.gz',
    },
  ],
};

function readPackageJson(rootDir) {
  return JSON.parse(fs.readFileSync(path.join(rootDir, 'package.json'), 'utf8'));
}

function readChecksumFile(rootDir) {
  return JSON.parse(fs.readFileSync(path.join(rootDir, CHECKSUMS_FILE), 'utf8'));
}

function resolveReleasePin(rootDir) {
  const packageTag = readPackageJson(rootDir)?.llamacpp?.runtimeReleaseTag;
  const checksums = readChecksumFile(rootDir);
  if (typeof packageTag !== 'string' || !packageTag.trim()) {
    throw new Error('package.json is missing llamacpp.runtimeReleaseTag.');
  }
  if (checksums.version !== packageTag.trim()) {
    throw new Error(
      `llama.cpp bundle checksums are for ${checksums.version}, but package.json pins ${packageTag.trim()}.`,
    );
  }
  if (typeof checksums.repository !== 'string' || !checksums.repository.includes('/')) {
    throw new Error('llama.cpp bundle checksums are missing the official repository.');
  }
  return {
    version: packageTag.trim(),
    repository: checksums.repository,
    sha256: checksums.sha256 || {},
  };
}

function bundleEntriesFor(platform, arch, releaseTag) {
  const templates = BUNDLE_TARGETS[`${platform}/${arch}`] || [];
  return templates.map(entry => ({
    backend: entry.backend,
    accelerator: entry.accelerator,
    assetName: entry.assetTemplate.replaceAll('{tag}', releaseTag),
  }));
}

function officialAssetUrl(pin, assetName) {
  return `https://github.com/${pin.repository}/releases/download/${pin.version}/${assetName}`;
}

function executableNameFor(platform) {
  return platform === 'win32' ? 'llama-server.exe' : 'llama-server';
}

function bundledVersionDir(outputRoot, version) {
  return path.join(outputRoot, BUNDLED_DIRECTORY_NAME, version);
}

async function bundleLlamaCppBackends(options) {
  const rootDir = options.rootDir;
  const platform = options.platform;
  const arch = options.arch;
  const pin = resolveReleasePin(rootDir);
  const outputRoot = options.outputRoot || path.join(rootDir, 'resources', 'llamacpp-backends');
  const entries = bundleEntriesFor(platform, arch, pin.version);
  const versionDir = bundledVersionDir(outputRoot, pin.version);

  fs.rmSync(path.join(outputRoot, BUNDLED_DIRECTORY_NAME), { recursive: true, force: true });
  if (entries.length === 0) {
    console.log(
      `[llamacpp-bundle] ${platform}/${arch} has no bundled llama.cpp backend; CUDA stays on demand.`,
    );
    return [];
  }

  fs.mkdirSync(versionDir, { recursive: true });
  const executableName = executableNameFor(platform);
  for (const entry of entries) {
    const expectedSha256 = pin.sha256[entry.assetName];
    if (typeof expectedSha256 !== 'string' || !/^[a-f0-9]{64}$/.test(expectedSha256)) {
      throw new Error(`Missing pinned SHA-256 for ${entry.assetName}.`);
    }
    const destinationDir = path.join(versionDir, entry.backend);
    await installOfficialBackend({
      pin,
      entry,
      expectedSha256,
      destinationDir,
      platform,
      arch,
      executableName,
    });
  }
  return entries;
}

async function installOfficialBackend(options) {
  const { pin, entry, expectedSha256, destinationDir, platform, arch, executableName } = options;
  const url = officialAssetUrl(pin, entry.assetName);
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'llamacpp-bundle-'));
  try {
    const archivePath = path.join(tempDir, entry.assetName);
    console.log(`[llamacpp-bundle] Downloading ${url}`);
    await downloadVerified(url, archivePath, expectedSha256);
    const extractDir = path.join(tempDir, 'extract');
    fs.mkdirSync(extractDir, { recursive: true });
    await extractArchive(archivePath, extractDir);
    const executablePath = findExecutablePath(extractDir, executableName);
    if (!executablePath) {
      throw new Error(`${entry.assetName} does not contain ${executableName}.`);
    }
    fs.rmSync(destinationDir, { recursive: true, force: true });
    const binDir = path.join(destinationDir, 'bin');
    fs.mkdirSync(binDir, { recursive: true });
    copyDirectoryContents(path.dirname(executablePath), binDir);
    removeModelWeights(destinationDir);
    const installedExecutable = path.join(binDir, executableName);
    if (!fs.existsSync(installedExecutable)) {
      throw new Error(`Installed backend is missing ${executableName}.`);
    }
    if (platform !== 'win32') fs.chmodSync(installedExecutable, 0o755);
    fs.writeFileSync(
      path.join(destinationDir, BUILD_INFO_FILE),
      `${JSON.stringify(
        {
          schemaVersion: 1,
          version: pin.version,
          backend: entry.backend,
          target: entry.backend,
          accelerator: entry.accelerator,
          platform,
          arch,
          source: 'bundled',
          assetName: entry.assetName,
          sha256: expectedSha256,
          releaseUrl: url,
        },
        null,
        2,
      )}\n`,
      'utf8',
    );
    console.log(`[llamacpp-bundle] Installed ${entry.backend} from ${entry.assetName}`);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

async function downloadVerified(url, destination, expectedSha256) {
  const response = await fetch(url, {
    headers: { 'User-Agent': 'ZhiYuanAgent/llamacpp-backend-bundler' },
    redirect: 'follow',
  });
  if (!response.ok || !response.body) {
    throw new Error(`Download failed: HTTP ${response.status} ${response.statusText} ${url}`);
  }
  await fs.promises.mkdir(path.dirname(destination), { recursive: true });
  const hash = crypto.createHash('sha256');
  const file = fs.createWriteStream(destination);
  const reader = response.body.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      hash.update(value);
      if (!file.write(Buffer.from(value))) {
        await new Promise(resolve => file.once('drain', resolve));
      }
    }
  } catch (error) {
    file.destroy();
    throw error;
  }
  await new Promise((resolve, reject) => {
    file.end(finishError => (finishError ? reject(finishError) : resolve()));
  });
  const digest = hash.digest('hex');
  if (digest !== expectedSha256.toLowerCase()) {
    await fs.promises.rm(destination, { force: true });
    throw new Error(
      `Checksum mismatch for ${path.basename(destination)}: expected ${expectedSha256}, got ${digest}.`,
    );
  }
}

async function extractArchive(archivePath, extractDir) {
  if (archivePath.endsWith('.tar.gz')) {
    const tar = require('tar');
    await tar.x({ file: archivePath, cwd: extractDir });
    return;
  }
  if (archivePath.endsWith('.zip')) {
    const extractZip = require('extract-zip');
    await extractZip(archivePath, { dir: extractDir });
    return;
  }
  throw new Error(`Unsupported llama.cpp archive: ${path.basename(archivePath)}`);
}

function findExecutablePath(rootDir, executableName) {
  const queue = [rootDir];
  while (queue.length > 0) {
    const currentDir = queue.shift();
    if (!currentDir) continue;
    const directExecutable = path.join(currentDir, executableName);
    if (fs.existsSync(directExecutable) && fs.statSync(directExecutable).isFile()) {
      return directExecutable;
    }
    for (const entry of fs.readdirSync(currentDir, { withFileTypes: true })) {
      if (entry.isDirectory()) queue.push(path.join(currentDir, entry.name));
    }
  }
  return null;
}

function copyDirectoryContents(sourceDir, targetDir) {
  for (const entry of fs.readdirSync(sourceDir, { withFileTypes: true })) {
    const sourcePath = path.join(sourceDir, entry.name);
    const targetPath = path.join(targetDir, entry.name);
    if (entry.isSymbolicLink()) {
      fs.symlinkSync(fs.readlinkSync(sourcePath), targetPath);
      continue;
    }
    if (entry.isDirectory()) {
      fs.mkdirSync(targetPath, { recursive: true });
      copyDirectoryContents(sourcePath, targetPath);
      continue;
    }
    if (entry.isFile()) fs.copyFileSync(sourcePath, targetPath);
  }
}

function removeModelWeights(rootDir) {
  const weights = findModelWeights(rootDir);
  for (const weight of weights) fs.rmSync(weight, { force: true });
  if (findModelWeights(rootDir).length > 0) {
    throw new Error(`Failed to remove llama.cpp model weights from ${rootDir}.`);
  }
}

function findModelWeights(rootDir) {
  const weights = [];
  const queue = [rootDir];
  while (queue.length > 0) {
    const currentDir = queue.shift();
    if (!currentDir || !fs.existsSync(currentDir)) continue;
    for (const entry of fs.readdirSync(currentDir, { withFileTypes: true })) {
      const nextPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        queue.push(nextPath);
        continue;
      }
      if (entry.isFile() && entry.name.toLowerCase().endsWith('.gguf')) weights.push(nextPath);
    }
  }
  return weights;
}

function verifyBundledLlamaCpp(options) {
  const platform = options.platform;
  const arch = options.arch;
  const releaseTag = options.releaseTag;
  const resourcesDir = options.resourcesDir;
  const entries = bundleEntriesFor(platform, arch, releaseTag);
  const bundledRoot = path.join(resourcesDir, 'llamacpp-backends', BUNDLED_DIRECTORY_NAME);
  if (entries.length === 0) {
    if (fs.existsSync(bundledRoot)) {
      throw new Error(
        `llama.cpp binaries were bundled for ${platform}/${arch}, which has no bundled backend.`,
      );
    }
    return [];
  }
  if (!fs.existsSync(bundledRoot)) {
    throw new Error(`Bundled llama.cpp backends are missing from ${bundledRoot}.`);
  }
  const versions = fs.readdirSync(bundledRoot);
  if (versions.length !== 1 || versions[0] !== releaseTag) {
    throw new Error(
      `Bundled llama.cpp version must be ${releaseTag} only, found ${versions.join(', ') || 'nothing'}.`,
    );
  }
  const versionDir = path.join(bundledRoot, releaseTag);
  const present = new Set(fs.readdirSync(versionDir));
  const expected = new Set(entries.map(entry => entry.backend));
  for (const name of present) {
    if (!expected.has(name)) {
      throw new Error(`Unexpected bundled llama.cpp backend for ${platform}/${arch}: ${name}.`);
    }
  }
  const executableName = executableNameFor(platform);
  const outputs = [];
  for (const entry of entries) {
    const backendDir = path.join(versionDir, entry.backend);
    const executablePath = path.join(backendDir, 'bin', executableName);
    if (!fs.existsSync(executablePath) || !fs.statSync(executablePath).isFile()) {
      throw new Error(`Bundled llama.cpp backend ${entry.backend} is missing ${executableName}.`);
    }
    if (fs.statSync(executablePath).size <= 0) {
      throw new Error(`Bundled llama.cpp backend ${entry.backend} has an empty ${executableName}.`);
    }
    const buildInfo = JSON.parse(fs.readFileSync(path.join(backendDir, BUILD_INFO_FILE), 'utf8'));
    if (
      buildInfo.version !== releaseTag ||
      buildInfo.backend !== entry.backend ||
      buildInfo.accelerator !== entry.accelerator ||
      buildInfo.platform !== platform ||
      buildInfo.arch !== arch ||
      buildInfo.source !== 'bundled'
    ) {
      throw new Error(`Bundled llama.cpp metadata does not match ${entry.backend}.`);
    }
    const weights = findModelWeights(backendDir);
    if (weights.length > 0) {
      throw new Error(
        `Bundled llama.cpp backend ${entry.backend} contains model weights: ${weights.join(', ')}.`,
      );
    }
    const hostPlatform = options.hostPlatform ?? process.platform;
    const hostArch = options.hostArch ?? process.arch;
    if (hostPlatform === platform && hostArch === arch) {
      const output = (options.runVersion || runLlamaCppVersion)(executablePath, platform);
      if (!String(output || '').trim()) {
        throw new Error(`${executablePath} --version produced no output.`);
      }
      outputs.push(String(output).trim());
    }
  }
  return outputs;
}

function runLlamaCppVersion(executablePath, platform) {
  const binDir = path.dirname(executablePath);
  const env = { ...process.env };
  if (platform === 'linux') {
    env.LD_LIBRARY_PATH = joinPathEntry(binDir, env.LD_LIBRARY_PATH);
  } else if (platform === 'darwin') {
    env.DYLD_LIBRARY_PATH = joinPathEntry(binDir, env.DYLD_LIBRARY_PATH);
  } else if (platform === 'win32') {
    env.PATH = joinPathEntry(binDir, env.PATH);
  }
  const result = spawnSync(executablePath, ['--version'], {
    encoding: 'utf8',
    env,
    timeout: 30_000,
    windowsHide: true,
  });
  if (result.error) {
    throw new Error(`${executablePath} --version failed: ${result.error.message}`);
  }
  if (result.status !== 0) {
    throw new Error(
      `${path.basename(executablePath)} --version exited ${result.status}: ${result.stderr || result.stdout || ''}`.trim(),
    );
  }
  return `${result.stdout || ''}\n${result.stderr || ''}`.trim();
}

function joinPathEntry(entry, current) {
  return current ? `${entry}${path.delimiter}${current}` : entry;
}

function parseArgs(argv) {
  const options = { platform: process.platform, arch: process.arch, verify: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--verify') options.verify = true;
    else if (arg === '--platform') options.platform = argv[++index];
    else if (arg === '--arch') options.arch = argv[++index];
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return options;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const rootDir = path.resolve(__dirname, '..');
  const pin = resolveReleasePin(rootDir);
  if (options.verify) {
    const outputs = verifyBundledLlamaCpp({
      resourcesDir: path.join(rootDir, 'resources'),
      platform: options.platform,
      arch: options.arch,
      releaseTag: pin.version,
    });
    for (const output of outputs) console.log(`[llamacpp-bundle] ${output}`);
    return;
  }
  await bundleLlamaCppBackends({
    rootDir,
    platform: options.platform,
    arch: options.arch,
  });
  if (bundleEntriesFor(options.platform, options.arch, pin.version).length === 0) return;
  const outputs = verifyBundledLlamaCpp({
    resourcesDir: path.join(rootDir, 'resources'),
    platform: options.platform,
    arch: options.arch,
    releaseTag: pin.version,
    hostPlatform: process.platform,
    hostArch: process.arch,
  });
  for (const output of outputs) console.log(`[llamacpp-bundle] ${output}`);
}

if (require.main === module) {
  main().catch(error => {
    console.error(`[llamacpp-bundle] ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  });
}

module.exports = {
  BUNDLED_DIRECTORY_NAME,
  bundleEntriesFor,
  bundleLlamaCppBackends,
  officialAssetUrl,
  resolveReleasePin,
  verifyBundledLlamaCpp,
};
