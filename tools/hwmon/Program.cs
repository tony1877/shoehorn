// shoehorn-hwmon: reads the temperature and voltage sensors through LibreHardwareMonitorLib every 2 s and serves them as
// /data.json on 127.0.0.1 (port 8085, or the first argument), in the tree shape of LibreHardwareMonitor's own web
// server: root > computer > hardware > sensors, each sensor with SensorId, Type, Text and Value. Sub-hardware sensors
// are listed under their hardware. Only temperatures and voltages are served; nothing listens beyond this machine.
// The root also carries "Updated" (Unix ms of the snapshot) so a reader can tell fresh data from a stalled sampler.

using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Runtime.InteropServices;
using System.Text.Json;
using LibreHardwareMonitor.Hardware;

// A driver update or reset can fault inside the sensor libraries (NVML), which .NET cannot catch. So this process is
// a supervisor: it runs itself with --worker as a child and starts it again when it exits. No crash dialog for either
// (the error mode is inherited).
SetErrorMode(0x0001 | 0x0002);  // SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX
if (!args.Contains("--worker"))
{
    string self = Environment.ProcessPath!;
    while (true)
    {
        var psi = new ProcessStartInfo(self) { UseShellExecute = false };
        psi.ArgumentList.Add("--worker");
        psi.ArgumentList.Add(Environment.ProcessId.ToString(CultureInfo.InvariantCulture));
        foreach (string a in args) psi.ArgumentList.Add(a);
        using var child = Process.Start(psi);
        child?.WaitForExit();
        Thread.Sleep(3000);
    }
}

// worker arguments: --worker <supervisor pid> [port]
int workerAt = Array.IndexOf(args, "--worker");
Process? supervisor = null;
try
{
    supervisor = Process.GetProcessById(int.Parse(args[workerAt + 1], CultureInfo.InvariantCulture));
}
catch { }
string? portArg = args.Where((_, i) => i != workerAt && i != workerAt + 1).FirstOrDefault();
int port = portArg != null && int.TryParse(portArg, out var p) ? p : 8085;

var computer = new Computer
{
    IsCpuEnabled = true,
    IsGpuEnabled = true,
    IsMemoryEnabled = true,
    IsMotherboardEnabled = true,
    IsStorageEnabled = true,
    IsControllerEnabled = true,
};
// Sampling can hang inside a sensor library (a driver call that never returns): the old snapshot would then be served
// forever. A watchdog kills this worker when no sample has finished for StallMs, so the supervisor starts a fresh one;
// the hardware that was updating at the time is logged and skipped for SkipMinutes, so one bad device cannot keep the
// rest stale. The log and the skip list live in %ProgramData%\shoehorn-hwmon.
const int StallMs = 20_000, FirstSampleMs = 90_000, SkipMinutes = 30;
string stateDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData), "shoehorn-hwmon");
Directory.CreateDirectory(stateDir);
string logPath = Path.Combine(stateDir, "hwmon.log"), skipPath = Path.Combine(stateDir, "skip.txt");

void Log(string line)
{
    try
    {
        if (File.Exists(logPath) && new FileInfo(logPath).Length > 256 * 1024) File.Move(logPath, logPath + ".1", true);
        File.AppendAllText(logPath, $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} [{Environment.ProcessId}] {line}{Environment.NewLine}");
    }
    catch { }
}

// identifier -> skip until (UTC)
Dictionary<string, DateTime> LoadSkips()
{
    var skips = new Dictionary<string, DateTime>();
    try
    {
        foreach (string l in File.Exists(skipPath) ? File.ReadAllLines(skipPath) : [])
        {
            string[] f = l.Split('\t');
            if (f.Length == 2 && long.TryParse(f[1], out long until) && DateTimeOffset.FromUnixTimeSeconds(until).UtcDateTime > DateTime.UtcNow)
                skips[f[0]] = DateTimeOffset.FromUnixTimeSeconds(until).UtcDateTime;
        }
    }
    catch { }
    return skips;
}
var skips = LoadSkips();
foreach (var (id, until) in skips) Log($"skipping {id} until {until.ToLocalTime():HH:mm} (it stalled sampling)");

long lastSample = Environment.TickCount64;  // the watchdog's clock: end of the last finished sample, or start
bool sampledOnce = false;
string current = "opening the sensor libraries";  // what the sampler is doing right now, for the stall log
string currentDevice = "";  // top-level hardware being updated (a hung sub-device quarantines its parent)
new Thread(() =>
{
    while (true)
    {
        Thread.Sleep(1000);
        // a worker whose supervisor is gone (the task was ended) must not keep the port from the next one
        if (supervisor == null || supervisor.HasExited) Process.GetCurrentProcess().Kill();
        long age = Environment.TickCount64 - Interlocked.Read(ref lastSample);
        if (age < (Volatile.Read(ref sampledOnce) ? StallMs : FirstSampleMs)) continue;
        string what = Volatile.Read(ref current), device = Volatile.Read(ref currentDevice);
        Log($"sampling stalled for {age / 1000} s in: {what}; restarting{(device != "" ? $", skipping {device} for {SkipMinutes} min" : "")}");
        if (device != "")
        {
            try
            {
                var next = LoadSkips();
                next[device] = DateTime.UtcNow.AddMinutes(SkipMinutes);
                File.WriteAllLines(skipPath, next.Select(kv => $"{kv.Key}\t{new DateTimeOffset(kv.Value).ToUnixTimeSeconds()}"));
            }
            catch { }
        }
        // TerminateProcess: a thread stuck in a driver call would keep a graceful exit waiting
        Process.GetCurrentProcess().Kill();
    }
}) { IsBackground = true, Name = "watchdog" }.Start();

computer.Open();

// Board voltage maps LibreHardwareMonitorLib lacks: sensor name -> (new name, factor on the library's value; 0 = drop).
// ROG CROSSHAIR X670E GENE (NCT6799D): the library lists the inputs raw ("This is wrong" on Vcore and VTT). The
// dividers of ASUS's ROG CROSSHAIR X870E HERO / APEX map (same input layout, matched to HWiNFO in the library) fit
// it: the 1.8 V PLL reads 1.80 V and both chipsets their nominal 1.05 V.
var boardMaps = new Dictionary<string, Dictionary<string, (string Name, double Factor)>>
{
    ["ROG CROSSHAIR X670E GENE"] = new()
    {
        ["Vcore"] = ("Vcore", 151.0 / 136 / 2),  // the library already doubled it
        ["Voltage #6"] = ("CPU VDD_MISC", 91.0 / 82),
        ["Voltage #7"] = ("CPU SoC", 91.0 / 82),
        ["CPU Termination"] = ("", 0),
        ["Voltage #11"] = ("Chipset 1 VDD", 2),
        ["Voltage #12"] = ("Chipset 2 VDD", 2),
        ["Voltage #13"] = ("Chipset Standby", 1),
        ["Voltage #14"] = ("CPU VDDIO Memory", 91.0 / 82),
        ["Voltage #15"] = ("1.8V PLL", 53.0 / 36),
    },
};

byte[] body = "{}"u8.ToArray();
var gate = new object();

void Sample()
{
    var hardware = new List<object>();
    var skipped = new List<string>();
    foreach (IHardware hw in computer.Hardware)
    {
        string hwId = hw.Identifier.ToString();
        if (skips.TryGetValue(hwId, out DateTime until) && until > DateTime.UtcNow)
        {
            skipped.Add(Clean(hw.Name));
            continue;
        }
        Volatile.Write(ref currentDevice, hwId);
        Volatile.Write(ref current, hwId);
        hw.Update();
        var sensors = new List<object>();
        void Add(IHardware h)
        {
            foreach (ISensor s in h.Sensors)
            {
                if (s.SensorType is not (SensorType.Temperature or SensorType.Voltage) || s.Value is not float v) continue;
                bool volt = s.SensorType == SensorType.Voltage;
                string name = s.Name;
                var map = volt ? boardMaps.FirstOrDefault(b => Clean(hw.Name).Contains(b.Key, StringComparison.OrdinalIgnoreCase)).Value : null;
                if (map != null && map.TryGetValue(name, out var m))
                {
                    if (m.Factor == 0) continue;
                    name = m.Name;
                    v = (float) (v * m.Factor);
                }
                sensors.Add(new Dictionary<string, object>
                {
                    ["Text"] = name,
                    ["SensorId"] = s.Identifier.ToString(),
                    ["Type"] = volt ? "Voltage" : "Temperature",
                    ["Value"] = v.ToString(volt ? "0.000" : "0.0", CultureInfo.InvariantCulture) + (volt ? " V" : " C"),
                });
            }
        }
        Add(hw);
        foreach (IHardware sub in hw.SubHardware)
        {
            Volatile.Write(ref current, sub.Identifier.ToString());
            sub.Update();
            Add(sub);
        }
        hardware.Add(new Dictionary<string, object> { ["Text"] = Clean(hw.Name), ["Children"] = sensors });
    }
    Volatile.Write(ref currentDevice, "");
    Volatile.Write(ref current, "serializing");
    var root = new Dictionary<string, object>
    {
        ["Text"] = "Sensor",
        ["Updated"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
        ["Skipped"] = skipped,
        ["Children"] = new[] { new Dictionary<string, object> { ["Text"] = Environment.MachineName, ["Children"] = hardware } },
    };
    byte[] json = JsonSerializer.SerializeToUtf8Bytes(root);
    lock (gate) body = json;
}

// names read from hardware (a memory module's SPD part number) can carry unprintable bytes
static string Clean(string s) =>
    System.Text.RegularExpressions.Regex.Replace(new string(s.Where(c => c >= ' ' && c <= '~').ToArray()), " {2,}", " ").Trim();

// One sampler thread: a sample never overlaps the previous one (the sensor libraries are not thread-safe), and a slow
// one delays the next instead of piling up.
new Thread(() =>
{
    while (true)
    {
        long t0 = Environment.TickCount64;
        try
        {
            Sample();
        }
        catch (Exception e)
        {
            Log($"sample failed in {Volatile.Read(ref current)}: {e.GetType().Name}: {e.Message}");
        }
        Interlocked.Exchange(ref lastSample, Environment.TickCount64);
        Volatile.Write(ref sampledOnce, true);
        Volatile.Write(ref current, "idle");
        Thread.Sleep((int) Math.Max(100, 2000 - (Environment.TickCount64 - t0)));
    }
}) { IsBackground = true, Name = "sampler" }.Start();

var listener = new HttpListener();
listener.Prefixes.Add($"http://127.0.0.1:{port}/");
listener.Start();
while (true)
{
    HttpListenerContext ctx = listener.GetContext();
    try
    {
        byte[] b;
        lock (gate) b = body;
        if (ctx.Request.Url?.AbsolutePath == "/data.json")
        {
            ctx.Response.ContentType = "application/json";
            ctx.Response.OutputStream.Write(b);
        }
        else
        {
            ctx.Response.StatusCode = 404;
        }
    }
    catch { }
    finally { ctx.Response.Close(); }
}

[DllImport("kernel32.dll")]
static extern uint SetErrorMode(uint mode);
