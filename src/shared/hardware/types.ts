export type NvidiaGpuInfo = {
  index: number;
  name: string;
  memoryTotalMiB: number;
  memoryFreeMiB?: number;
  driverVersion?: string;
};

export type NvidiaSmiSnapshot = {
  source: 'nvidia-smi';
  available: boolean;
  checkedAt: string;
  gpus: NvidiaGpuInfo[];
  /** Driver version from the first GPU row, when available. */
  driverVersion?: string;
  error?: string;
};

export type SystemMemorySnapshot = {
  source: 'system';
  available: boolean;
  checkedAt: string;
  totalMemoryMiB: number;
  freeMemoryMiB: number;
  error?: string;
};
