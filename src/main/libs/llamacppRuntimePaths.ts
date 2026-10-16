import { execFile } from 'child_process';
import { app } from 'electron';
import fs from 'fs';
import path from 'path';
import { promisify } from 'util';

import type {
  LlamaCppRuntimeBackend as LlamaCppRuntimeBackendType,
  LlamaCppRuntimeCudaMajor as LlamaCppRuntimeCudaMajorType,
  LlamaCppServiceConfig,
  LlamaCppStatusSnapshot,
} from '../../shared/llamacpp';
import { LlamaCppRuntimeBackend, LlamaCppRuntimeCudaMajor } from '../../shared/llamacpp';
import { resolveBundledLlamaCppBackendDir } from './llamacppBackendResources';
import {
  LLAMACPP_BUNDLED_BACKEND_DIRECTORY,
  resolvePreferredLlamaCppExecutable,
} from './llamacppBundledBackend';
import { isPathInside } from './llamacppModelCatalog';
import { LlamaCppRuntimeTargetId } from './llamacppRuntimeConstants';
import { resolveLlamaCppRuntimeTargetId } from './llamacppRuntimeInstaller';
export { prependEnvPathEntry, resolveExecutableDir } from './llamacppProcessRuntimePaths';

const execFileAsync = promisify(execFile);

export async function findLlamaCppExecutable(
  config: LlamaCppServiceConfig = {},
): Promise<string | null> {
  const envPath = process.env.LLAMACPP_BIN?.trim();
  if (envPath && fs.existsSync(envPath)) return envPath;

  try {
    const preferred = await resolvePreferredLlamaCppExecutable({
      platform: process.platform,
      arch: process.arch,
      bundledRoot: path.join(
        resolveBundledLlamaCppBackendDir(),
        LLAMACPP_BUNDLED_BACKEND_DIRECTORY,
      ),
      userRuntimeRoot: getUserLlamaCppRuntimeRoot(),
      desiredBackend: config.runtimeBackend,
    });
    if (preferred && fs.existsSync(preferred.executablePath)) return preferred.executablePath;
  } catch (error) {
    console.warn('[LlamaCpp] bundled backend lookup failed:', error);
  }

  for (const candidate of buildLlamaCppExecutableCandidates({
    platform: process.platform,
    isPackaged: app.isPackaged,
    resourceRoot: process.resourcesPath || path.join(__dirname, '..', '..'),
    appRoot: path.join(__dirname, '..', '..'),
    cwd: process.cwd(),
    userRuntimeRoot: getUserLlamaCppRuntimeRoot(),
    envPath: process.env.LLAMACPP_BIN,
  })) {
    if (fs.existsSync(candidate)) return candidate;
  }

  if (app.isPackaged) {
    return null;
  }

  const command = process.platform === 'win32' ? 'where' : 'which';
  try {
    const { stdout } = await execFileAsync(command, ['llama-server'], { timeout: 1000 });
    const first = stdout
      .split(/\r?\n/)
      .map(line => line.trim())
      .find(Boolean);
    return first || null;
  } catch {
    return null;
  }
}

export async function findExternalLlamaCppExecutable(
  _config: LlamaCppServiceConfig = {},
): Promise<string | null> {
  for (const candidate of buildLlamaCppExecutableCandidates({
    platform: process.platform,
    isPackaged: app.isPackaged,
    resourceRoot: process.resourcesPath || path.join(__dirname, '..', '..'),
    appRoot: path.join(__dirname, '..', '..'),
    cwd: process.cwd(),
    userRuntimeRoot: getUserLlamaCppRuntimeRoot(),
    envPath: process.env.LLAMACPP_BIN,
  })) {
    if (!fs.existsSync(candidate)) continue;
    if (isPathInside(candidate, getUserLlamaCppRuntimeRoot())) continue;
    return candidate;
  }
  return null;
}

export function buildLlamaCppExecutableCandidates(input: {
  platform: NodeJS.Platform;
  isPackaged: boolean;
  resourceRoot: string;
  appRoot: string;
  cwd: string;
  userRuntimeRoot: string;
  envPath?: string;
}): string[] {
  const extension = input.platform === 'win32' ? '.exe' : '';
  const join = (...segments: string[]) =>
    (input.platform === 'win32'
      ? path.win32.join(...segments)
      : path.posix.join(...segments)
    ).replace(/\\/g, '/');
  const candidates = [
    input.envPath?.trim(),
    join(input.userRuntimeRoot, 'current', 'build', 'bin', `llama-server${extension}`),
    join(input.userRuntimeRoot, 'current', 'bin', `llama-server${extension}`),
    join(input.userRuntimeRoot, 'current', `llama-server${extension}`),
    join(input.resourceRoot, 'llamacpp', `llama-server${extension}`),
    join(input.resourceRoot, 'llamacpp', 'bin', `llama-server${extension}`),
  ];

  if (!input.isPackaged) {
    candidates.push(
      join(input.appRoot, 'vendor', 'llamacpp-runtime', 'current', `llama-server${extension}`),
      join(
        input.appRoot,
        'vendor',
        'llamacpp-runtime',
        'current',
        'build',
        'bin',
        `llama-server${extension}`,
      ),
      join(
        input.appRoot,
        'vendor',
        'llamacpp-runtime',
        'current',
        'bin',
        `llama-server${extension}`,
      ),
      join(input.cwd, 'vendor', 'llamacpp-runtime', 'current', `llama-server${extension}`),
      join(
        input.cwd,
        'vendor',
        'llamacpp-runtime',
        'current',
        'build',
        'bin',
        `llama-server${extension}`,
      ),
      join(input.cwd, 'vendor', 'llamacpp-runtime', 'current', 'bin', `llama-server${extension}`),
      '/opt/homebrew/bin/llama-server',
      '/usr/local/bin/llama-server',
      '/usr/bin/llama-server',
    );
  }

  return Array.from(
    new Set(candidates.filter((candidate): candidate is string => Boolean(candidate))),
  );
}

export function getUserLlamaCppRuntimeRoot(): string {
  return path.join(app.getPath('userData'), 'llamacpp-runtime');
}

export function resolveLlamaCppRuntimeTargetPreference(config: LlamaCppServiceConfig): {
  runtimeBackend: LlamaCppRuntimeBackendType;
  runtimeCudaMajor: LlamaCppRuntimeCudaMajorType;
} {
  return {
    runtimeBackend: config.runtimeBackend ?? LlamaCppRuntimeBackend.Auto,
    runtimeCudaMajor: config.runtimeCudaMajor ?? LlamaCppRuntimeCudaMajor.Cuda12,
  };
}

export function selectLlamaCppRuntimeTarget(input: {
  platform: NodeJS.Platform;
  arch: string;
  runtimeBackend: LlamaCppRuntimeBackendType;
  runtimeCudaMajor: LlamaCppRuntimeCudaMajorType;
  hasNvidiaGpu: boolean;
}): { ok: true; targetId: string } | { ok: false; error: string } {
  const baseTargetId = resolveLlamaCppRuntimeTargetId(input.platform, input.arch);
  if (!baseTargetId) {
    return {
      ok: false,
      error: `Unsupported platform for llama.cpp runtime: ${input.platform}/${input.arch}.`,
    };
  }

  if (input.platform !== 'win32') {
    return { ok: true, targetId: baseTargetId };
  }

  if (baseTargetId !== LlamaCppRuntimeTargetId.WinX64) {
    if (input.runtimeBackend === LlamaCppRuntimeBackend.Cuda) {
      return {
        ok: false,
        error: 'CUDA runtime is only supported on Windows x64.',
      };
    }
    return { ok: true, targetId: baseTargetId };
  }

  if (input.runtimeBackend === LlamaCppRuntimeBackend.Cpu) {
    return { ok: true, targetId: LlamaCppRuntimeTargetId.WinX64 };
  }
  if (input.runtimeBackend === LlamaCppRuntimeBackend.Cuda) {
    if (!input.hasNvidiaGpu) {
      return { ok: false, error: 'CUDA runtime requires an NVIDIA GPU on Windows.' };
    }
    return { ok: true, targetId: LlamaCppRuntimeTargetId.WinX64Cuda12 };
  }

  return {
    ok: true,
    targetId: input.hasNvidiaGpu
      ? LlamaCppRuntimeTargetId.WinX64Cuda12
      : LlamaCppRuntimeTargetId.WinX64,
  };
}

export function resolveLlamaCppRuntimeMetadata(
  executablePath: string | undefined,
): Partial<LlamaCppStatusSnapshot> {
  if (!executablePath) {
    return {
      runtimeTargetId: undefined,
      runtimeBackend: undefined,
      runtimeCudaMajor: undefined,
      runtimeRoot: undefined,
      deviceProbeAvailable: false,
    };
  }
  const runtimeRoot = getManagedRuntimeRootForExecutable(executablePath);
  const buildInfo = runtimeRoot ? readRuntimeBuildMetadata(runtimeRoot) : undefined;
  const targetId = buildInfo?.target;
  const version = buildInfo?.version;
  const backend = buildInfo?.backend ?? targetId;
  return {
    ...(version ? { runtimeVersion: version } : {}),
    ...(backend ? { runtimeBackendId: backend } : {}),
    ...(version && backend ? { versionBackend: `${version}/${backend}` } : {}),
    ...(buildInfo?.source ? { runtimeSource: buildInfo.source } : {}),
    ...(targetId ? { runtimeTargetId: targetId } : {}),
    ...runtimeBackendFieldsFromTargetId(targetId),
    ...(runtimeRoot ? { runtimeRoot } : {}),
    deviceProbeAvailable: true,
  };
}

function getManagedRuntimeRootForExecutable(executablePath: string): string | undefined {
  const userRuntimeRoot = getUserLlamaCppRuntimeRoot();
  const userCurrentRoot = path.join(userRuntimeRoot, 'current');
  if (isPathInside(executablePath, userCurrentRoot)) {
    return userCurrentRoot;
  }

  const cwdCurrentRoot = path.join(process.cwd(), 'vendor', 'llamacpp-runtime', 'current');
  if (isPathInside(executablePath, cwdCurrentRoot)) {
    return cwdCurrentRoot;
  }
  return findAncestorRuntimeRoot(executablePath);
}

function findAncestorRuntimeRoot(executablePath: string): string | undefined {
  let current = path.dirname(path.resolve(executablePath));
  for (let depth = 0; depth < 4; depth += 1) {
    if (fs.existsSync(path.join(current, 'runtime-build-info.json'))) return current;
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return undefined;
}

function readRuntimeBuildMetadata(runtimeRoot: string):
  | {
      target?: string;
      backend?: string;
      version?: string;
      source?: string;
    }
  | undefined {
  const buildInfoPath = path.join(runtimeRoot, 'runtime-build-info.json');
  try {
    const buildInfo = JSON.parse(fs.readFileSync(buildInfoPath, 'utf-8')) as {
      target?: string;
      targetId?: string;
      backend?: string;
      version?: string;
      source?: string;
    };
    const target =
      buildInfo.target?.trim() || buildInfo.targetId?.trim() || buildInfo.backend?.trim();
    return {
      ...(target ? { target } : {}),
      ...(buildInfo.backend?.trim() ? { backend: buildInfo.backend.trim() } : {}),
      ...(buildInfo.version?.trim() ? { version: buildInfo.version.trim() } : {}),
      ...(buildInfo.source?.trim() ? { source: buildInfo.source.trim() } : {}),
    };
  } catch {
    return undefined;
  }
}

function runtimeBackendFieldsFromTargetId(
  targetId: string | undefined,
): Pick<Partial<LlamaCppStatusSnapshot>, 'runtimeBackend' | 'runtimeCudaMajor'> {
  if (targetId === LlamaCppRuntimeTargetId.WinX64Cuda12) {
    return {
      runtimeBackend: LlamaCppRuntimeBackend.Cuda,
      runtimeCudaMajor: LlamaCppRuntimeCudaMajor.Cuda12,
    };
  }
  if (targetId?.includes('cuda')) {
    return { runtimeBackend: LlamaCppRuntimeBackend.Cuda };
  }
  if (targetId) {
    return { runtimeBackend: LlamaCppRuntimeBackend.Cpu };
  }
  return {};
}
