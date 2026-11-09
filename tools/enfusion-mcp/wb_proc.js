// wb_proc.js PID | --env - print one tab-separated line for a Windows process, read-only (ticket DD-14a, CC-05):
//   ProcessId <TAB> ExecutablePath <TAB> CreationDate <TAB> CommandLine
// from WMI Win32_Process, through cscript.exe (Windows Script Host), so the endpoint handshake needs no PowerShell.
// It only queries WMI and writes to stdout. Exit codes: 0 one process found; 1 none; 2 usage.
var args = WScript.Arguments;
// --env: print the TEMP and TMP this process sees (the handshake checks they are its data-drive dir before any query).
if (args.length === 1 && args(0) === "--env") {
    var penv = WScript.CreateObject("WScript.Shell").Environment("PROCESS");
    WScript.StdOut.WriteLine("TEMP=" + penv("TEMP") + "\tTMP=" + penv("TMP"));
    WScript.Quit(0);
}
if (args.length !== 1 || !/^[0-9]{1,10}$/.test(args(0))) {
    WScript.StdErr.WriteLine("usage: cscript //NoLogo wb_proc.js PID");
    WScript.Quit(2);
}
var svc = GetObject("winmgmts:{impersonationLevel=impersonate}!\\\\.\\root\\cimv2");
var query = "SELECT ProcessId, ExecutablePath, CreationDate, CommandLine FROM Win32_Process WHERE ProcessId = " + args(0);
var items = new Enumerator(svc.ExecQuery(query));
var n = 0;
for (; !items.atEnd(); items.moveNext()) {
    var p = items.item();
    n++;
    WScript.StdOut.WriteLine(p.ProcessId + "\t" + (p.ExecutablePath || "") + "\t" + (p.CreationDate || "") + "\t" +
                             (p.CommandLine || ""));
}
WScript.Quit(n === 1 ? 0 : 1);
