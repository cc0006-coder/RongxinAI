import { spawnSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { afterEach, expect, test } from 'vitest';

import { LLAMACPP_BUNDLED_BACKEND_DIRECTORY } from '../src/main/libs/llamacppBundledBackend';

const { BUNDLED_DIRECTORY_NAME, bundleEntriesFor, resolveReleasePin, verifyBundledLlamaCpp } =
  require('./bundle-llamacpp-backends.cjs') as {
    BUNDLED_DIRECTORY_NAME: string;
    bundleEntriesFor: (
      platform: string,
      arch: string,
      releaseTag: string,
    ) => Array<{ backend: string; accelerator: string; assetName: string }>;
    resolveReleasePin: (rootDir: string) => {
      version: string;
      repository: string;
      sha256: Record<string, string>;
    };
    verifyBundledLlamaCpp: (options: {
      resourcesDir: string;
      platform: string;
      arch: string;
      releaseTag: string;
      hostPlatform?: string;
      hostArch?: string;
      runVersion?: (executablePath: string, platform: string) => string;
    }) => string[];
  };

const rootDir = path.resolve(__dirname, '..');
const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test('pins official cpu, vulkan, and metal archives without cuda or model weights', () => {
  const pin = resolveReleasePin(rootDir);
  expect(pin.version).toBe('b9244');
  expect(pin.repository).toBe('ggml-org/llama.cpp');
  expect(BUNDLED_DIRECTORY_NAME).toBe(LLAMACPP_BUNDLED_BACKEND_DIRECTORY);

  const entries = [
    ...bundleEntriesFor('win32', 'x64', pin.version),
    ...bundleEntriesFor('linux', 'x64', pin.version),
    ...bundleEntriesFor('linux', 'arm64', pin.version),
    ...bundleEntriesFor('darwin', 'arm64', pin.version),
    ...bundleEntriesFor('darwin', 'x64', pin.version),
  ];

  expect(bundleEntriesFor('win32', 'arm64', pin.version)).toEqual([]);
  expect(entries.map(entry => entry.assetName)).toEqual([
    'llama-b9244-bin-win-cpu-x64.zip',
    'llama-b9244-bin-win-vulkan-x64.zip',
    'llama-b9244-bin-ubuntu-x64.tar.gz',
    'llama-b9244-bin-ubuntu-vulkan-x64.tar.gz',
    'llama-b9244-bin-ubuntu-arm64.tar.gz',
    'llama-b9244-bin-ubuntu-vulkan-arm64.tar.gz',
    'llama-b9244-bin-macos-arm64.tar.gz',
    'llama-b9244-bin-macos-x64.tar.gz',
  ]);
  expect(
    entries.some(entry => entry.assetName.includes('cuda') || entry.assetName.includes('gguf')),
  ).toBe(false);
  for (const entry of entries) {
    expect(pin.sha256[entry.assetName]).toMatch(/^[a-f0-9]{64}$/);
  }
  expect(bundleEntriesFor('darwin', 'arm64', pin.version)[0]?.accelerator).toBe('metal');
  expect(bundleEntriesFor('linux', 'x64', pin.version).map(entry => entry.accelerator)).toEqual([
    'cpu',
    'vulkan',
  ]);
});

test('packaging check requires the bundled binaries and runs --version', () => {
  const resourcesDir = createTemporaryDirectory();
  const calls: string[] = [];
  writePackagedBackend(resourcesDir, 'linux-x64', 'cpu');
  writePackagedBackend(resourcesDir, 'linux-x64-vulkan', 'vulkan');

  const outputs = verifyBundledLlamaCpp({
    resourcesDir,
    platform: 'linux',
    arch: 'x64',
    releaseTag: 'b9244',
    hostPlatform: 'linux',
    hostArch: 'x64',
    runVersion: executablePath => {
      calls.push(executablePath);
      const result = spawnSync(executablePath, ['--version'], { encoding: 'utf8' });
      expect(result.status).toBe(0);
      return `${result.stdout}${result.stderr}`;
    },
  });

  expect(calls).toHaveLength(2);
  expect(outputs.every(output => output.includes('b9244'))).toBe(true);
});

test('packaging check rejects model weights and missing binaries', () => {
  const resourcesDir = createTemporaryDirectory();
  writePackagedBackend(resourcesDir, 'linux-x64', 'cpu');
  fs.writeFileSync(
    path.join(
      resourcesDir,
      'llamacpp-backends',
      'bundled',
      'b9244',
      'linux-x64',
      'bin',
      'model.gguf',
    ),
    'weights',
  );

  expect(() =>
    verifyBundledLlamaCpp({
      resourcesDir,
      platform: 'linux',
      arch: 'x64',
      releaseTag: 'b9244',
      hostPlatform: 'win32',
      hostArch: 'x64',
    }),
  ).toThrow('model weights');

  const missingDir = createTemporaryDirectory();
  writePackagedBackend(missingDir, 'linux-x64', 'cpu');
  fs.rmSync(
    path.join(
      missingDir,
      'llamacpp-backends',
      'bundled',
      'b9244',
      'linux-x64',
      'bin',
      'llama-server',
    ),
  );
  expect(() =>
    verifyBundledLlamaCpp({
      resourcesDir: missingDir,
      platform: 'linux',
      arch: 'x64',
      releaseTag: 'b9244',
      hostPlatform: 'win32',
    }),
  ).toThrow('missing llama-server');
});

function createTemporaryDirectory(): string {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'llamacpp-package-'));
  temporaryDirectories.push(directory);
  return directory;
}

function writePackagedBackend(resourcesDir: string, backend: string, accelerator: string): void {
  const binDir = path.join(resourcesDir, 'llamacpp-backends', 'bundled', 'b9244', backend, 'bin');
  fs.mkdirSync(binDir, { recursive: true });
  const executablePath = path.join(binDir, 'llama-server');
  fs.writeFileSync(
    executablePath,
    '#!/bin/sh\nif [ "$1" = "--version" ]; then echo "version: b9244"; exit 0; fi\nexit 1\n',
  );
  fs.chmodSync(executablePath, 0o755);
  fs.writeFileSync(
    path.join(binDir, '..', 'runtime-build-info.json'),
    JSON.stringify({
      version: 'b9244',
      backend,
      accelerator,
      platform: 'linux',
      arch: 'x64',
      source: 'bundled',
    }),
  );
}
