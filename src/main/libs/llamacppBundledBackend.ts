import { execFile } from 'child_process';
import fs from 'fs';
import path from 'path';
import { promisify } from 'util';

import type {
  LlamaCppBackendAccelerator,
  LlamaCppBackendInfo,
  LlamaCppBackendRef,
  LlamaCppRuntimeBackend,
} from '../../shared/llamacpp';
import { LlamaCppRuntimeBackend as LlamaCppRuntimeBackendValue } from '../../shared/llamacpp';
import {
  buildLlamaCppServeEnv,
  isGpuLikeRuntimeDevice,
  parseLlamaCppListDevicesOutput,
} from './llamacppServe';

export const LLAMACPP_BUNDLED_BACKEND_DIRECTORY = 'bundled';
const BUILD_INFO_FILE = 'runtime-build-info.json';
const gpuProbeCache = new Map<string, boolean>();
const execFileAsync = promisify(execFile);

export type LlamaCppResolvedBackend = {
  version: string;
  backend: string;
  versionBackend: string;
  accelerator: LlamaCppBackendAccelerator;
  platform: string;
  arch: string;
  executablePath: string;
  directory: string;
  source: 'bundled' | 'downloaded';
};

type BackendList = {
  backends: LlamaCppBackendInfo[];
  selection?: LlamaCppBackendRef;
  recommended?: LlamaCppBackendRef;
};

export function listBundledLlamaCppBackends(bundledRoot: string): LlamaCppResolvedBackend[] {
  return listBackendDirectories(bundledRoot, 'bundled');
}

export function listDownloadedCudaBackends(userRuntimeRoot: string): LlamaCppResolvedBackend[] {
  return listBackendDirectories(path.join(userRuntimeRoot, 'backends'), 'downloaded').filter(
    backend => backend.accelerator === 'cuda',
  );
}

export function resolveLlamaCppBackendPreference(input: {
  bundled: LlamaCppResolvedBackend[];
  downloaded: LlamaCppResolvedBackend[];
  platform: NodeJS.Platform;
  arch: string;
  hasUsableGpu: boolean;
  desiredBackend?: LlamaCppRuntimeBackend;
}): LlamaCppResolvedBackend | undefined {
  const bundled = input.bundled.filter(backend =>
    matchesMachine(backend, input.platform, input.arch),
  );
  const downloaded = input.downloaded.filter(backend =>
    matchesMachine(backend, input.platform, input.arch),
  );
  const desired = input.desiredBackend ?? LlamaCppRuntimeBackendValue.Auto;
  if (desired === LlamaCppRuntimeBackendValue.Cpu) {
    return findAccelerator(bundled, 'cpu') ?? findAccelerator(bundled, 'metal');
  }
  if (desired === LlamaCppRuntimeBackendValue.Cuda) {
    return bestDownloadedCuda(downloaded);
  }
  const cuda = bestDownloadedCuda(downloaded);
  if (cuda) return cuda;
  if (input.hasUsableGpu) {
    const vulkan = findAccelerator(bundled, 'vulkan');
    if (vulkan) return vulkan;
  }
  return findAccelerator(bundled, 'cpu') ?? findAccelerator(bundled, 'metal');
}

export async function resolvePreferredLlamaCppExecutable(input: {
  platform: NodeJS.Platform;
  arch: string;
  bundledRoot: string;
  userRuntimeRoot: string;
  desiredBackend?: LlamaCppRuntimeBackend;
  hasUsableGpu?: boolean;
  probeGpu?: (executablePath: string) => Promise<string>;
}): Promise<LlamaCppResolvedBackend | undefined> {
  const bundled = listBundledLlamaCppBackends(input.bundledRoot);
  const downloaded = listDownloadedCudaBackends(input.userRuntimeRoot);
  const desired = input.desiredBackend ?? LlamaCppRuntimeBackendValue.Auto;
  let hasUsableGpu = input.hasUsableGpu;
  if (hasUsableGpu === undefined) {
    hasUsableGpu = false;
    if (desired === LlamaCppRuntimeBackendValue.Auto) {
      const vulkan = bundled.find(
        backend =>
          backend.accelerator === 'vulkan' && matchesMachine(backend, input.platform, input.arch),
      );
      if (vulkan) {
        hasUsableGpu = await detectUsableLlamaCppGpu({
          executablePath: vulkan.executablePath,
          probe: input.probeGpu,
        });
      }
    }
  }
  return resolveLlamaCppBackendPreference({
    bundled,
    downloaded,
    platform: input.platform,
    arch: input.arch,
    hasUsableGpu,
    desiredBackend: desired,
  });
}

export function shouldActivateResolvedBackend(
  preferred: LlamaCppResolvedBackend | undefined,
  desiredBackend: LlamaCppRuntimeBackend | undefined,
): preferred is LlamaCppResolvedBackend {
  if (!preferred) return false;
  if ((desiredBackend ?? LlamaCppRuntimeBackendValue.Auto) === LlamaCppRuntimeBackendValue.Cuda) {
    return preferred.accelerator === 'cuda';
  }
  return true;
}

export function findBundledLlamaCppBackend(
  bundledRoot: string,
  ref: Pick<LlamaCppBackendRef, 'version' | 'backend'>,
  platform: NodeJS.Platform,
  arch: string,
): LlamaCppResolvedBackend | undefined {
  return listBundledLlamaCppBackends(bundledRoot).find(
    backend =>
      backend.version === ref.version &&
      backend.backend === ref.backend &&
      matchesMachine(backend, platform, arch),
  );
}

export function linkBundledBackendIntoRuntime(
  userRuntimeRoot: string,
  backend: LlamaCppResolvedBackend,
): void {
  const backendDir = path.join(userRuntimeRoot, 'backends', backend.version, backend.backend);
  fs.mkdirSync(path.dirname(backendDir), { recursive: true });
  const target = path.resolve(backend.directory);
  try {
    const stat = fs.lstatSync(backendDir);
    if (stat.isSymbolicLink()) {
      const currentTarget = path.resolve(path.dirname(backendDir), fs.readlinkSync(backendDir));
      if (currentTarget === target) return;
      fs.unlinkSync(backendDir);
    } else {
      fs.rmSync(backendDir, { recursive: true, force: true });
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
  }
  fs.symlinkSync(target, backendDir, process.platform === 'win32' ? 'junction' : 'dir');
}

export function applyLlamaCppBackendPreference<T extends BackendList>(
  list: T,
  bundled: LlamaCppResolvedBackend[],
  preferred: LlamaCppResolvedBackend | undefined,
  machine: { platform: NodeJS.Platform; arch: string },
): T {
  const backends = list.backends.map(backend => ({ ...backend }));
  for (const item of bundled) {
    if (!matchesMachine(item, machine.platform, machine.arch)) continue;
    const existing = backends.find(
      backend => backend.version === item.version && backend.backend === item.backend,
    );
    if (existing) {
      existing.installed = true;
      continue;
    }
    backends.push({
      version: item.version,
      backend: item.backend,
      versionBackend: item.versionBackend,
      platform: normalizePlatform(item.platform),
      arch: normalizeArch(item.arch),
      accelerator: item.accelerator,
      installed: true,
      recommended: false,
      current: list.selection?.versionBackend === item.versionBackend,
      source: 'bundled',
    });
  }
  const recommended = preferred
    ? toBackendRef(preferred.version, preferred.backend)
    : list.recommended;
  if (preferred) {
    for (const backend of backends) {
      backend.recommended =
        backend.version === preferred.version && backend.backend === preferred.backend;
    }
  }
  return {
    ...list,
    backends,
    recommended,
  };
}

export async function detectUsableLlamaCppGpu(input: {
  executablePath?: string;
  probe?: (executablePath: string) => Promise<string>;
}): Promise<boolean> {
  const executablePath = input.executablePath;
  if (!executablePath || !fs.existsSync(executablePath)) return false;
  const cached = gpuProbeCache.get(executablePath);
  if (cached !== undefined && !input.probe) return cached;
  try {
    const output = input.probe
      ? await input.probe(executablePath)
      : await probeLlamaCppDevices(executablePath);
    const usable = parseLlamaCppListDevicesOutput(output).some(device =>
      isGpuLikeRuntimeDevice(device),
    );
    if (!input.probe) gpuProbeCache.set(executablePath, usable);
    return usable;
  } catch {
    if (!input.probe) gpuProbeCache.set(executablePath, false);
    return false;
  }
}

export function clearLlamaCppGpuProbeCache(): void {
  gpuProbeCache.clear();
}

async function probeLlamaCppDevices(executablePath: string): Promise<string> {
  const { stdout, stderr } = await execFileAsync(executablePath, ['--list-devices'], {
    env: buildLlamaCppServeEnv(process.env, executablePath),
    encoding: 'utf8',
    timeout: 10_000,
    windowsHide: true,
    maxBuffer: 256 * 1024,
  });
  return [stdout, stderr].filter(Boolean).join('\n');
}

function listBackendDirectories(
  rootDir: string,
  source: LlamaCppResolvedBackend['source'],
): LlamaCppResolvedBackend[] {
  if (!fs.existsSync(rootDir)) return [];
  const backends: LlamaCppResolvedBackend[] = [];
  for (const versionEntry of fs.readdirSync(rootDir, { withFileTypes: true })) {
    if (!versionEntry.isDirectory()) continue;
    const versionDir = path.join(rootDir, versionEntry.name);
    for (const backendEntry of fs.readdirSync(versionDir, { withFileTypes: true })) {
      if (!backendEntry.isDirectory()) continue;
      const directory = path.join(versionDir, backendEntry.name);
      const resolved = readResolvedBackend(directory, source);
      if (resolved) backends.push(resolved);
    }
  }
  return backends;
}

function readResolvedBackend(
  directory: string,
  source: LlamaCppResolvedBackend['source'],
): LlamaCppResolvedBackend | undefined {
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(fs.readFileSync(path.join(directory, BUILD_INFO_FILE), 'utf8')) as Record<
      string,
      unknown
    >;
  } catch {
    return undefined;
  }
  const version = stringValue(parsed.version);
  const backend = stringValue(parsed.backend) || stringValue(parsed.target);
  const platform = normalizePlatform(stringValue(parsed.platform) || inferPlatform(backend));
  const arch = normalizeArch(stringValue(parsed.arch) || inferArch(backend));
  const accelerator = normalizeAccelerator(
    stringValue(parsed.accelerator) || inferAccelerator(backend),
  );
  if (!version || !backend || !accelerator) return undefined;
  const executablePath = findLlamaServerExecutable(directory, platform);
  if (!executablePath) return undefined;
  return {
    version,
    backend,
    versionBackend: `${version}/${backend}`,
    accelerator,
    platform,
    arch,
    executablePath,
    directory,
    source,
  };
}

function findLlamaServerExecutable(directory: string, platform: string): string | undefined {
  const executableName = platform === 'win32' ? 'llama-server.exe' : 'llama-server';
  const candidates = [
    path.join(directory, 'build', 'bin', executableName),
    path.join(directory, 'bin', executableName),
    path.join(directory, executableName),
  ];
  return candidates.find(candidate => fs.existsSync(candidate) && fs.statSync(candidate).isFile());
}

function findAccelerator(
  backends: LlamaCppResolvedBackend[],
  accelerator: LlamaCppBackendAccelerator,
): LlamaCppResolvedBackend | undefined {
  return backends.find(backend => backend.accelerator === accelerator);
}

function bestDownloadedCuda(
  backends: LlamaCppResolvedBackend[],
): LlamaCppResolvedBackend | undefined {
  const cuda = backends.filter(backend => backend.accelerator === 'cuda');
  cuda.sort((left, right) => {
    const versionOrder = compareVersionDesc(left.version, right.version);
    if (versionOrder !== 0) return versionOrder;
    return cudaMajor(right.backend) - cudaMajor(left.backend);
  });
  return cuda[0];
}

function compareVersionDesc(left: string, right: string): number {
  return (
    Number.parseInt(right.replace(/^b/, '') || '0', 10) -
    Number.parseInt(left.replace(/^b/, '') || '0', 10)
  );
}

function cudaMajor(backend: string): number {
  const match = backend.match(/cuda-(\d+)/);
  return match ? Number.parseInt(match[1], 10) : 0;
}

function matchesMachine(
  backend: LlamaCppResolvedBackend,
  platform: NodeJS.Platform,
  arch: string,
): boolean {
  return (
    normalizePlatform(backend.platform) === normalizePlatform(platform) &&
    normalizeArch(backend.arch) === normalizeArch(arch)
  );
}

function toBackendRef(version: string, backend: string): LlamaCppBackendRef {
  return { version, backend, versionBackend: `${version}/${backend}` };
}

function normalizePlatform(platform: string): string {
  if (platform === 'windows' || platform === 'win') return 'win32';
  if (platform === 'macos' || platform === 'mac') return 'darwin';
  return platform;
}

function normalizeArch(arch: string): string {
  if (arch === 'amd64') return 'x64';
  if (arch === 'aarch64') return 'arm64';
  return arch;
}

function inferPlatform(backend: string): string {
  if (backend.startsWith('win-')) return 'win32';
  if (backend.startsWith('mac-') || backend.startsWith('macos-')) return 'darwin';
  if (backend.startsWith('linux-') || backend.startsWith('ubuntu-')) return 'linux';
  return '';
}

function inferArch(backend: string): string {
  if (backend.endsWith('-arm64')) return 'arm64';
  if (backend.endsWith('-x64')) return 'x64';
  return '';
}

function inferAccelerator(backend: string): string {
  if (backend.includes('cuda')) return 'cuda';
  if (backend.includes('vulkan')) return 'vulkan';
  if (backend.startsWith('mac-') || backend.startsWith('macos-')) return 'metal';
  if (backend.includes('cpu') || backend.startsWith('win-') || backend.startsWith('linux-')) {
    return 'cpu';
  }
  return '';
}

function normalizeAccelerator(value: string): LlamaCppBackendAccelerator | undefined {
  if (
    value === 'cpu' ||
    value === 'cuda' ||
    value === 'metal' ||
    value === 'vulkan' ||
    value === 'hip' ||
    value === 'openvino' ||
    value === 'sycl' ||
    value === 'unknown'
  ) {
    return value;
  }
  return undefined;
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}
