// Supervises the Python sidecar child process from the Electron main process.
// Spawns it, reads the SIDECAR_PORT=<n> line it prints on startup (spec §4.4),
// and reports unexpected exits. spawnFn is injectable so this is unit-testable
// without launching a real process.

export interface SpawnedLike {
  stdout: { on(event: 'data', cb: (chunk: Buffer) => void): void } | null
  stderr: { on(event: 'data', cb: (chunk: Buffer) => void): void } | null
  on(event: 'exit', cb: (code: number | null) => void): void
  kill(signal?: string): void
}

export interface SupervisorOptions {
  spawnFn: (command: string, args: string[], options: { cwd?: string }) => SpawnedLike
  pythonPath: string
  scriptPath: string
  cwd?: string
  onPort?: (port: number) => void
  onExit?: (code: number | null) => void
}

const PORT_RE = /SIDECAR_PORT=(\d+)/

export class SidecarSupervisor {
  private opts: SupervisorOptions
  private child: SpawnedLike | null = null
  private stdoutBuf = ''
  private stopped = false
  private portReported = false

  constructor(opts: SupervisorOptions) {
    this.opts = opts
  }

  start(): void {
    this.stopped = false
    this.portReported = false
    this.stdoutBuf = ''
    const { spawnFn, pythonPath, scriptPath, cwd } = this.opts
    const child = spawnFn(pythonPath, [scriptPath], { cwd })
    this.child = child

    child.stdout?.on('data', (chunk) => this.onStdout(chunk.toString()))
    child.on('exit', (code) => {
      if (!this.stopped) this.opts.onExit?.(code)
    })
  }

  private onStdout(text: string): void {
    this.stdoutBuf += text
    let idx: number
    while ((idx = this.stdoutBuf.indexOf('\n')) !== -1) {
      const line = this.stdoutBuf.slice(0, idx)
      this.stdoutBuf = this.stdoutBuf.slice(idx + 1)
      if (!this.portReported) {
        const m = line.match(PORT_RE)
        if (m) {
          this.portReported = true
          this.opts.onPort?.(Number(m[1]))
        }
      }
    }
  }

  stop(): void {
    this.stopped = true
    this.child?.kill()
    this.child = null
  }
}
