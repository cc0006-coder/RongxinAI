import fs from 'fs';
import os from 'os';
import path from 'path';

import { afterEach, expect, test } from 'vitest';

import { LlamaCppRuntimeBackend } from '../../shared/llamacpp';
import {
  applyLlamaCppBackendPreference,
  clearLlamaCppGpuProbeCache,
  linkBundledBackendIntoRuntime,
  listBundledLlamaCppBackends,
  resolveLlamaCppBackendPreference,
  resolvePreferredLlamaCppExecutable,
} from './llamacppBundledBackend';

const temporaryDirectories: string[] = [];

afterEach(() => {
  clearLlamaCppGpuProbeCache();
  for (const directory of temporaryDirectories.splice(0)) {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test('resolver finds the bundled vulkan backend when a gpu is usable', () => {
  const bundledRoot = createTemporaryDirectory();
  const cpu = writeBackend(bundledRoot, {
    version: 'b9244',
    backend: 'linux-x64',
    accelerator: 'cpu',
    platform: 'linux',
    arch: 'x64',
  });
  const vulkan = writeBackend(bundledRoot, {
    version: 'b9244',
    backend: 'linux-x64-vulkan',
    accelerator: 'vulkan',
    platform: 'linux',
    arch: 'x64',
  });

  const bundled = listBundledLlamaCppBackends(bundledRoot);
  const selected = resolveLlamaCppBackendPreference({
    bundled,
    downloaded: [],
    platform: 'linux',
    arch: 'x64',
    hasUsableGpu: true,
  });

  expect(bundled.map(backend => backend.executablePath).sort()).toEqual(
    [cpu.executablePath, vulkan.executablePath].sort(),
  );
  expect(selected?.executablePath).toBe(vulkan.executablePath);
  expect(fs.existsSync(selected?.executablePath ?? '')).toBe(true);
});

test('resolver uses the bundled cpu backend when no gpu is usable', () => {
  const bundledRoot = createTemporaryDirectory();
  const cpu = writeBackend(bundledRoot, {
    version: 'b9244',
    backend: 'win-x64',
    accelerator: 'cpu',
    platform: 'win32',
    arch: 'x64',
    executableName: 'llama-server.exe',
  });
  writeBackend(bundledRoot, {
    version: 'b9244',
    backend: 'win-x64-vulkan',
    accelerator: 'vulkan',
    platform: 'win32',
    arch: 'x64',
    executableName: 'llama-server.exe',
  });

  const selected = resolveLlamaCppBackendPreference({
    bundled: listBundledLlamaCppBackends(bundledRoot),
    downloaded: [],
    platform: 'win32',
    arch: 'x64',
    hasUsableGpu: false,
  });

  expect(selected?.executablePath).toBe(cpu.executablePath);
});

test('resolver prefers a downloaded cuda backend over bundled builds', () => {
  const bundledRoot = createTemporaryDirectory();
  const runtimeRoot = createTemporaryDirectory();
  writeBackend(bundledRoot, {
    version: 'b9244',
    backend: 'win-x64-vulkan',
    accelerator: 'vulkan',
    platform: 'win32',
    arch: 'x64',
    executableName: 'llama-server.exe',
  });
  const cuda = writeBackend(path.join(runtimeRoot, 'backends'), {
    version: 'b9244',
    backend: 'win-x64-cuda-12',
    accelerator: 'cuda',
    platform: 'win32',
    arch: 'x64',
    executableName: 'llama-server.exe',
  });

  const selected = resolveLlamaCppBackendPreference({
    bundled: listBundledLlamaCppBackends(bundledRoot),
    downloaded: listBundledLlamaCppBackends(path.join(runtimeRoot, 'backends')).map(backend => ({
      ...backend,
      source: 'downloaded' as const,
    })),
    platform: 'win32',
    arch: 'x64',
    hasUsableGpu: true,
  });

  expect(selected?.accelerator).toBe('cuda');
  expect(selected?.executablePath).toBe(cuda.executablePath);
});

test('resolver keeps an explicit cpu choice on the bundled cpu backend', () => {
  const bundledRoot = createTemporaryDirectory();
  const cpu = writeBackend(bundledRoot, {
    version: 'b9244',
    backend: 'linux-x64',
    accelerator: 'cpu',
    platform: 'linux',
    arch: 'x64',
  });
  writeBackend(bundledRoot, {
    version: 'b9244',
    backend: 'linux-x64-vulkan',
    accelerator: 'vulkan',
    platform: 'linux',
    arch: 'x64',
  });
  const cuda = writeBackend(createTemporaryDirectory(), {
    version: 'b9244',
    backend: 'linux-x64-cuda',
    accelerator: 'cuda',
    platform: 'linux',
    arch: 'x64',
  });

  const selected = resolveLlamaCppBackendPreference({
    bundled: listBundledLlamaCppBackends(bundledRoot),
    downloaded: [{ ...cuda, source: 'downloaded' }],
    platform: 'linux',
    arch: 'x64',
    hasUsableGpu: true,
    desiredBackend: LlamaCppRuntimeBackend.Cpu,
  });

  expect(selected?.executablePath).toBe(cpu.executablePath);
});

test('resolver finds the bundled metal backend on macOS', () => {
  const bundledRoot = createTemporaryDirectory();
  const metal = writeBackend(bundledRoot, {
    version: 'b9244',
    backend: 'mac-arm64',
    accelerator: 'metal',
    platform: 'darwin',
    arch: 'arm64',
  });

  const selected = resolveLlamaCppBackendPreference({
    bundled: listBundledLlamaCppBackends(bundledRoot),
    downloaded: [],
    platform: 'darwin',
    arch: 'arm64',
    hasUsableGpu: false,
  });

  expect(selected?.executablePath).toBe(metal.executablePath);
  expect(selected?.accelerator).toBe('metal');
});

test('resolvePreferredLlamaCppExecutable probes the bundled vulkan backend', async () => {
  const bundledRoot = createTemporaryDirectory();
  writeBackend(bundledRoot, {
    version: 'b9244',
    backend: 'linux-x64',
    accelerator: 'cpu',
    platform: 'linux',
    arch: 'x64',
  });
  const vulkan = writeBackend(bundledRoot, {
    version: 'b9244',
    backend: 'linux-x64-vulkan',
    accelerator: 'vulkan',
    platform: 'linux',
    arch: 'x64',
  });

  const withGpu = await resolvePreferredLlamaCppExecutable({
    platform: 'linux',
    arch: 'x64',
    bundledRoot,
    userRuntimeRoot: createTemporaryDirectory(),
    probeGpu: async executablePath => {
      expect(executablePath).toBe(vulkan.executablePath);
      return 'Vulkan0: AMD Radeon';
    },
  });
  const withoutGpu = await resolvePreferredLlamaCppExecutable({
    platform: 'linux',
    arch: 'x64',
    bundledRoot,
    userRuntimeRoot: createTemporaryDirectory(),
    probeGpu: async () => 'CPU: CPU',
  });

  expect(withGpu?.backend).toBe('linux-x64-vulkan');
  expect(withoutGpu?.backend).toBe('linux-x64');
});

test('links a bundled backend into the user runtime without copying model files', () => {
  const bundledRoot = createTemporaryDirectory();
  const runtimeRoot = createTemporaryDirectory();
  const bundled = writeBackend(bundledRoot, {
    version: 'b9244',
    backend: 'linux-x64',
    accelerator: 'cpu',
    platform: 'linux',
    arch: 'x64',
  });

  linkBundledBackendIntoRuntime(runtimeRoot, bundled);
  const linked = path.join(runtimeRoot, 'backends', 'b9244', 'linux-x64');

  expect(fs.lstatSync(linked).isSymbolicLink()).toBe(true);
  expect(fs.existsSync(path.join(linked, 'bin', 'llama-server'))).toBe(true);
});

test('lists a bundled backend as installed and recommended ahead of on-demand cuda', () => {
  const bundledRoot = createTemporaryDirectory();
  const vulkan = writeBackend(bundledRoot, {
    version: 'b9244',
    backend: 'win-x64-vulkan',
    accelerator: 'vulkan',
    platform: 'win32',
    arch: 'x64',
    executableName: 'llama-server.exe',
  });
  const bundled = listBundledLlamaCppBackends(bundledRoot);
  const annotated = applyLlamaCppBackendPreference(
    {
      backends: [
        {
          version: 'b9244',
          backend: 'win-x64',
          versionBackend: 'b9244/win-x64',
          platform: 'win32',
          arch: 'x64',
          accelerator: 'cpu',
          installed: false,
          recommended: false,
          current: false,
          source: 'manifest',
        },
        {
          version: 'b9244',
          backend: 'win-x64-cuda-12',
          versionBackend: 'b9244/win-x64-cuda-12',
          platform: 'win32',
          arch: 'x64',
          accelerator: 'cuda',
          installed: false,
          recommended: true,
          current: false,
          source: 'manifest',
        },
      ],
      recommended: {
        version: 'b9244',
        backend: 'win-x64-cuda-12',
        versionBackend: 'b9244/win-x64-cuda-12',
      },
    },
    bundled,
    vulkan,
    { platform: 'win32', arch: 'x64' },
  );

  expect(annotated.recommended?.versionBackend).toBe('b9244/win-x64-vulkan');
  expect(annotated.backends.find(backend => backend.backend === 'win-x64-vulkan')).toMatchObject({
    installed: true,
    recommended: true,
    source: 'bundled',
  });
  expect(
    annotated.backends.find(backend => backend.backend === 'win-x64-cuda-12')?.recommended,
  ).toBe(false);
});

function createTemporaryDirectory(): string {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'llamacpp-bundled-'));
  temporaryDirectories.push(directory);
  return directory;
}

function writeBackend(
  rootDir: string,
  input: {
    version: string;
    backend: string;
    accelerator: string;
    platform: string;
    arch: string;
    executableName?: string;
  },
) {
  const directory = path.join(rootDir, input.version, input.backend);
  const executableName = input.executableName ?? 'llama-server';
  const executablePath = path.join(directory, 'bin', executableName);
  fs.mkdirSync(path.dirname(executablePath), { recursive: true });
  fs.writeFileSync(executablePath, 'binary');
  fs.writeFileSync(
    path.join(directory, 'runtime-build-info.json'),
    JSON.stringify({
      version: input.version,
      backend: input.backend,
      accelerator: input.accelerator,
      platform: input.platform,
      arch: input.arch,
      source: 'bundled',
    }),
  );
  return {
    version: input.version,
    backend: input.backend,
    versionBackend: `${input.version}/${input.backend}`,
    accelerator: input.accelerator as 'cpu',
    platform: input.platform,
    arch: input.arch,
    executablePath,
    directory,
    source: 'bundled' as const,
  };
}
