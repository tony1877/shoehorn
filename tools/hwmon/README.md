# shoehorn-hwmon

A windowless helper that reads the temperature and voltage sensors through
[LibreHardwareMonitorLib](https://github.com/LibreHardwareMonitor/LibreHardwareMonitor) (MPL-2.0) every 2 s and
serves them as `/data.json` on `127.0.0.1:8085`, the format of LibreHardwareMonitor's own web server, so
`shoehorn serve --hw-monitor` reads either. It needs administrator rights (CPU, memory module and board sensors go
through a kernel driver) and listens on this machine only.

```powershell
dotnet publish tools\hwmon -c Release -o C:\Tools\shoehorn-hwmon   # .NET 8 SDK
C:\Tools\shoehorn-hwmon\shoehorn-hwmon.exe [port]                 # asks for elevation
```

Start at logon without a prompt (run once in an elevated PowerShell):

```powershell
Register-ScheduledTask shoehorn-hwmon -RunLevel Highest -Trigger (New-ScheduledTaskTrigger -AtLogOn) `
    -Action (New-ScheduledTaskAction -Execute C:\Tools\shoehorn-hwmon\shoehorn-hwmon.exe) `
    -Settings (New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0 -AllowStartIfOnBatteries)
```

## When readings stop updating

A sensor library can hang inside a driver call. Before 2026-10-18 the helper then kept serving its last snapshot, and the
Sensors page showed hours-old values. Now:

- One sampler thread updates the sensors. A sample never overlaps the previous one.
- A watchdog kills the worker when no sample has finished for 20 s (90 s for the first one); the supervisor starts a new
  one 3 s later. The device that was updating is skipped for 30 min, so one stuck device cannot freeze the rest.
- The stall is written to `%ProgramData%\shoehorn-hwmon\hwmon.log`, and the skip list to `skip.txt` in the same folder.
- A worker exits when its supervisor is gone (the task was ended), so it cannot hold port 8085 from the next one.
- Each snapshot carries `Updated` (Unix ms) and `Skipped` (device names). `shoehorn serve` hides readings older than 10 s,
  or, from a server without the stamp (LibreHardwareMonitor), a body unchanged for 30 s. The Sensors page then says how
  stale they are.

To update a running install: `schtasks /End /TN shoehorn-hwmon`, publish over `C:\Tools\shoehorn-hwmon`,
`schtasks /Run /TN shoehorn-hwmon`.
