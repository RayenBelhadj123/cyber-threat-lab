using System.Diagnostics;
using System.Net.Http;
using System.Windows.Forms;

internal static class Program
{
    private const string StatusUrl = "http://localhost:15601/api/status";
    private const string DashboardUrl = "http://localhost:15601/app/dashboards#/view/cyber-overview";

    [STAThread]
    private static int Main(string[] args)
    {
        if (args.Contains("--check", StringComparer.OrdinalIgnoreCase))
        {
            return Uri.TryCreate(DashboardUrl, UriKind.Absolute, out var target) &&
                   target.Scheme == Uri.UriSchemeHttp ? 0 : 2;
        }

        ApplicationConfiguration.Initialize();
        Application.Run(new LauncherForm());
        return 0;
    }

    private static bool IsKibanaAvailable()
    {
        try
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(3) };
            using var response = client.GetAsync(StatusUrl).GetAwaiter().GetResult();
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    private static bool EnsureDockerEngineReady(Action<string>? reportStatus = null)
    {
        reportStatus?.Invoke("Checking Docker engine...");
        if (IsDockerEngineReady())
            return true;

        reportStatus?.Invoke("Starting Docker Desktop...");
        if (!StartDockerDesktop())
            return false;

        reportStatus?.Invoke("Waiting for Docker engine (up to 3 minutes)...");
        return WaitForDockerEngine(TimeSpan.FromMinutes(3));
    }

    private static bool WaitForDockerEngine(TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            if (IsDockerEngineReady())
                return true;
            Thread.Sleep(TimeSpan.FromSeconds(3));
        }
        return false;
    }

    private static bool StartDockerDesktop()
    {
        var programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
        var programFilesX86 = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);
        var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var candidates = new[]
        {
            Path.Combine(programFiles, "Docker", "Docker", "Docker Desktop.exe"),
            Path.Combine(programFilesX86, "Docker", "Docker", "Docker Desktop.exe"),
            Path.Combine(localAppData, "Programs", "Docker", "Docker", "Docker Desktop.exe")
        };
        var executable = candidates.FirstOrDefault(File.Exists);
        if (executable is null)
            return false;

        try
        {
            Process.Start(new ProcessStartInfo(executable) { UseShellExecute = true });
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool IsDockerEngineReady()
    {
        var start = new ProcessStartInfo("docker")
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        start.ArgumentList.Add("info");
        start.ArgumentList.Add("--format");
        start.ArgumentList.Add("{{.ServerVersion}}");
        return RunDockerProcess(start, TimeSpan.FromSeconds(10));
    }

    private static bool RunDockerProcess(ProcessStartInfo start, TimeSpan timeout)
    {
        try
        {
            using var process = Process.Start(start);
            if (process is null)
                return false;

            var stdout = process.StandardOutput.ReadToEndAsync();
            var stderr = process.StandardError.ReadToEndAsync();
            if (!process.WaitForExit((int)timeout.TotalMilliseconds))
            {
                try { process.Kill(entireProcessTree: true); } catch { }
                return false;
            }
            Task.WaitAll(stdout, stderr);
            return process.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }

    private static bool StartElasticServices()
    {
        var launcherDirectory = AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var labRoot = Directory.GetParent(launcherDirectory)?.FullName;
        if (labRoot is null || !File.Exists(Path.Combine(labRoot, "compose.yaml")))
            return false;

        var start = new ProcessStartInfo("docker")
        {
            WorkingDirectory = labRoot,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        start.ArgumentList.Add("compose");
        start.ArgumentList.Add("up");
        start.ArgumentList.Add("-d");
        start.ArgumentList.Add("elasticsearch");
        start.ArgumentList.Add("kibana");

        using var process = Process.Start(start);
        if (process is null)
            return false;

        var stdout = process.StandardOutput.ReadToEndAsync();
        var stderr = process.StandardError.ReadToEndAsync();
        if (!process.WaitForExit(120_000))
        {
            try { process.Kill(entireProcessTree: true); } catch { }
            return false;
        }
        Task.WaitAll(stdout, stderr);
        return process.ExitCode == 0;
    }

    private static bool WaitForKibana(TimeSpan timeout)
    {
        using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(3) };
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            try
            {
                using var response = client.GetAsync(StatusUrl).GetAwaiter().GetResult();
                if (response.IsSuccessStatusCode)
                    return true;
            }
            catch
            {
                // Kibana can take a short time to start after Compose returns.
            }
            Thread.Sleep(TimeSpan.FromSeconds(2));
        }
        return false;
    }

    private sealed class LauncherForm : Form
    {
        private readonly Label _status;
        private readonly ProgressBar _progress;

        public LauncherForm()
        {
            Text = "Cyber Threat Lab";
            StartPosition = FormStartPosition.CenterScreen;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            MinimizeBox = false;
            ShowInTaskbar = true;
            ClientSize = new System.Drawing.Size(440, 118);

            _status = new Label
            {
                AutoSize = false,
                Left = 18,
                Top = 18,
                Width = 404,
                Height = 32,
                Text = "Preparing Elastic dashboard...",
                TextAlign = System.Drawing.ContentAlignment.MiddleLeft
            };
            _progress = new ProgressBar
            {
                Left = 18,
                Top = 66,
                Width = 404,
                Height = 22,
                Style = ProgressBarStyle.Marquee,
                MarqueeAnimationSpeed = 24
            };
            Controls.Add(_status);
            Controls.Add(_progress);
            Shown += async (_, _) => await LaunchAsync();
        }

        private async Task LaunchAsync()
        {
            try
            {
                if (!await Task.Run(IsKibanaAvailable))
                {
                    if (!await Task.Run(() => EnsureDockerEngineReady(SetStatus)))
                        throw new InvalidOperationException("Docker Desktop could not be started or its engine did not become ready within 3 minutes.");

                    SetStatus("Starting Elasticsearch and Kibana...");
                    if (!await Task.Run(StartElasticServices))
                        throw new InvalidOperationException("Docker could not start Elasticsearch and Kibana. Keep this launcher in the project's dist folder and check Docker Desktop.");

                    SetStatus("Waiting for Kibana...");
                    if (!await Task.Run(() => WaitForKibana(TimeSpan.FromMinutes(3))))
                        throw new InvalidOperationException("Kibana did not become ready within 3 minutes. Check the Docker container logs and try again.");
                }

                SetStatus("Opening Kibana dashboard...");
                Process.Start(new ProcessStartInfo(DashboardUrl) { UseShellExecute = true });
                _progress.Style = ProgressBarStyle.Blocks;
                _progress.Value = 100;
                await Task.Delay(700);
                Close();
            }
            catch (Exception ex)
            {
                _progress.Style = ProgressBarStyle.Blocks;
                _progress.Value = 0;
                _status.Text = "Startup failed.";
                MessageBox.Show(this, ex.Message, "Cyber Threat Lab", MessageBoxButtons.OK, MessageBoxIcon.Error);
                Close();
            }
        }

        private void SetStatus(string text)
        {
            if (InvokeRequired)
                BeginInvoke(new Action(() => _status.Text = text));
            else
                _status.Text = text;
        }
    }
}
