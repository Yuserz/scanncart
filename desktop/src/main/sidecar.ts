// Supervises the Python sidecar child process from the Electron main process.
// Spawns it, reads the SIDECAR_PORT=<n> line it prints on startup (spec §4.4),
// and reports unexpected exits. spawnFn is injectable so this is unit-testable
// without launching a real process.

export interface SpawnedLike {
  stdout: { on(event: 'data', cb: (chunk: Buffer) => void): void } | null
  stderr: { on(event: 'data', cb: (chunk: Buffer) => void): void } | null
  on(event: 'exit', cb: (code: number | null) => void): void
  kill(): void
}

export interface SupervisorOptions {
  spawnFn: (
    command: string,
    args: string[],
    options: { cwd?: string; env?: NodeJS.ProcessEnv }
  ) => SpawnedLike
  pythonPath: string
  scriptPath: string
  cwd?: string
  onPort?: (port: number) => void
  onExit?: (code: number | null) => void
  onStderr?: (text: string) => void
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
    // Name our own pid so the sidecar can exit if we die without running
    // before-quit (a crash, or a force-kill). It cannot infer this itself:
    // .venv/Scripts/python.exe is a shim that re-execs the real interpreter,
    // so the sidecar's immediate parent is that shim, not us.
    const child = spawnFn(pythonPath, [scriptPath], {
      cwd,
      env: { ...process.env, SIDECAR_PARENT_PID: String(process.pid) }
    })
    this.child = child

    child.stdout?.on('data', (chunk) => this.onStdout(chunk.toString()))
    // Drain stderr too. uvicorn logs to stderr by default; if we never read it,
    // the OS pipe buffer fills and the sidecar blocks on its next write (frames
    // stop with no error). Attaching a 'data' listener keeps the pipe flowing.
    child.stderr?.on('data', (chunk) => this.opts.onStderr?.(chunk.toString()))
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
