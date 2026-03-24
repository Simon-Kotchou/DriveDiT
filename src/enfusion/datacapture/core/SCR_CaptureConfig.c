// ============================================================================
// SCR_CaptureConfig - Configuration System for Data Capture
// ============================================================================
//
// Centralized configuration management for the modular capture system.
// Supports hierarchical configuration with module-specific overrides.
//
// CONFIGURATION HIERARCHY:
// 1. Global defaults (built-in)
// 2. User configuration file ($profile:CaptureConfig.txt)
// 3. Runtime overrides (API calls)
// 4. Module-specific overrides
//
// ============================================================================

// -----------------------------------------------------------------------------
// Configuration keys (for type-safe access)
// -----------------------------------------------------------------------------
class SCR_ConfigKeys
{
    // Session
    static const string SESSION_BASE_PATH = "session.base_path";
    static const string SESSION_ID_FORMAT = "session.id_format";
    static const string SESSION_AUTO_FINALIZE = "session.auto_finalize";

    // Capture timing
    static const string CAPTURE_INTERVAL_MS = "capture.interval_ms";
    static const string CAPTURE_RATE_HZ = "capture.rate_hz";
    static const string CAPTURE_MAX_FRAMES = "capture.max_frames";
    static const string CAPTURE_MAX_DURATION_MS = "capture.max_duration_ms";

    // Output
    static const string OUTPUT_FORMAT = "output.format";
    static const string OUTPUT_COMPRESSION = "output.compression";
    static const string OUTPUT_SPLIT_SIZE_MB = "output.split_size_mb";

    // Buffer
    static const string BUFFER_CAPACITY = "buffer.capacity";
    static const string BUFFER_FLUSH_THRESHOLD = "buffer.flush_threshold";
    static const string BUFFER_FLUSH_INTERVAL_MS = "buffer.flush_interval_ms";
    static const string BUFFER_OVERFLOW_POLICY = "buffer.overflow_policy";

    // Logging
    static const string LOG_LEVEL = "log.level";
    static const string LOG_PROGRESS_INTERVAL = "log.progress_interval";
    static const string LOG_VERBOSE = "log.verbose";

    // Modules
    static const string MODULES_ENABLED = "modules.enabled";
    static const string MODULES_DISABLED = "modules.disabled";
}

// -----------------------------------------------------------------------------
// SCR_ConfigValue - Type-safe configuration value wrapper
// -----------------------------------------------------------------------------
class SCR_ConfigValue
{
    protected string m_sStringValue;
    protected float m_fFloatValue;
    protected int m_iIntValue;
    protected bool m_bBoolValue;
    protected int m_iType;  // 0=string, 1=float, 2=int, 3=bool

    //------------------------------------------------------------------------
    void SCR_ConfigValue()
    {
        m_sStringValue = "";
        m_fFloatValue = 0;
        m_iIntValue = 0;
        m_bBoolValue = false;
        m_iType = 0;
    }

    //------------------------------------------------------------------------
    static SCR_ConfigValue FromString(string value)
    {
        SCR_ConfigValue cv = new SCR_ConfigValue();
        cv.m_sStringValue = value;
        cv.m_iType = 0;
        return cv;
    }

    static SCR_ConfigValue FromFloat(float value)
    {
        SCR_ConfigValue cv = new SCR_ConfigValue();
        cv.m_fFloatValue = value;
        cv.m_sStringValue = value.ToString();
        cv.m_iType = 1;
        return cv;
    }

    static SCR_ConfigValue FromInt(int value)
    {
        SCR_ConfigValue cv = new SCR_ConfigValue();
        cv.m_iIntValue = value;
        cv.m_sStringValue = value.ToString();
        cv.m_iType = 2;
        return cv;
    }

    static SCR_ConfigValue FromBool(bool value)
    {
        SCR_ConfigValue cv = new SCR_ConfigValue();
        cv.m_bBoolValue = value;
        cv.m_sStringValue = value.ToString();
        cv.m_iType = 3;
        return cv;
    }

    //------------------------------------------------------------------------
    string AsString() { return m_sStringValue; }

    float AsFloat()
    {
        if (m_iType == 1)
            return m_fFloatValue;
        return m_sStringValue.ToFloat();
    }

    int AsInt()
    {
        if (m_iType == 2)
            return m_iIntValue;
        return m_sStringValue.ToInt();
    }

    bool AsBool()
    {
        if (m_iType == 3)
            return m_bBoolValue;
        return m_sStringValue == "true" || m_sStringValue == "1";
    }

    int GetType() { return m_iType; }
}

// -----------------------------------------------------------------------------
// SCR_ModuleConfig - Module-specific configuration
// -----------------------------------------------------------------------------
class SCR_ModuleConfig
{
    protected string m_sModuleId;
    protected bool m_bEnabled;
    protected float m_fIntervalMs;
    protected int m_iPriority;
    protected int m_iOutputFormats;
    protected ref map<string, ref SCR_ConfigValue> m_mSettings;

    //------------------------------------------------------------------------
    void SCR_ModuleConfig(string moduleId)
    {
        m_sModuleId = moduleId;
        m_bEnabled = true;
        m_fIntervalMs = 200;
        m_iPriority = 100;
        m_iOutputFormats = SCR_CaptureFormat.FORMAT_CSV;
        m_mSettings = new map<string, ref SCR_ConfigValue>();
    }

    //------------------------------------------------------------------------
    string GetModuleId() { return m_sModuleId; }

    bool IsEnabled() { return m_bEnabled; }
    void SetEnabled(bool enabled) { m_bEnabled = enabled; }

    float GetIntervalMs() { return m_fIntervalMs; }
    void SetIntervalMs(float intervalMs) { m_fIntervalMs = Math.Max(0, intervalMs); }

    int GetPriority() { return m_iPriority; }
    void SetPriority(int priority) { m_iPriority = priority; }

    int GetOutputFormats() { return m_iOutputFormats; }
    void SetOutputFormats(int formats) { m_iOutputFormats = formats; }

    //------------------------------------------------------------------------
    void SetSetting(string key, SCR_ConfigValue value)
    {
        m_mSettings.Set(key, value);
    }

    SCR_ConfigValue GetSetting(string key)
    {
        if (m_mSettings.Contains(key))
            return m_mSettings.Get(key);
        return null;
    }

    bool HasSetting(string key)
    {
        return m_mSettings.Contains(key);
    }

    //------------------------------------------------------------------------
    // Convenience setters/getters
    void SetStringValue(string key, string value)
    {
        SetSetting(key, SCR_ConfigValue.FromString(value));
    }

    void SetFloatValue(string key, float value)
    {
        SetSetting(key, SCR_ConfigValue.FromFloat(value));
    }

    void SetIntValue(string key, int value)
    {
        SetSetting(key, SCR_ConfigValue.FromInt(value));
    }

    void SetBoolValue(string key, bool value)
    {
        SetSetting(key, SCR_ConfigValue.FromBool(value));
    }

    string GetStringValue(string key, string defaultValue = "")
    {
        SCR_ConfigValue cv = GetSetting(key);
        if (cv)
            return cv.AsString();
        return defaultValue;
    }

    float GetFloatValue(string key, float defaultValue = 0)
    {
        SCR_ConfigValue cv = GetSetting(key);
        if (cv)
            return cv.AsFloat();
        return defaultValue;
    }

    int GetIntValue(string key, int defaultValue = 0)
    {
        SCR_ConfigValue cv = GetSetting(key);
        if (cv)
            return cv.AsInt();
        return defaultValue;
    }

    bool GetBoolValue(string key, bool defaultValue = false)
    {
        SCR_ConfigValue cv = GetSetting(key);
        if (cv)
            return cv.AsBool();
        return defaultValue;
    }
}

// -----------------------------------------------------------------------------
// SCR_CaptureConfig - Main configuration class
// -----------------------------------------------------------------------------
class SCR_CaptureConfig
{
    // Global settings
    protected ref map<string, ref SCR_ConfigValue> m_mGlobalSettings;

    // Module-specific settings
    protected ref map<string, ref SCR_ModuleConfig> m_mModuleConfigs;

    // Runtime state
    protected bool m_bLoaded;
    protected string m_sConfigPath;
    protected float m_fLastModifiedTime;

    // Defaults
    static const string DEFAULT_BASE_PATH = "$profile:DrivingData";
    static const float DEFAULT_CAPTURE_INTERVAL_MS = 200.0;
    static const int DEFAULT_BUFFER_CAPACITY = 1024;
    static const int DEFAULT_LOG_PROGRESS_INTERVAL = 500;

    //------------------------------------------------------------------------
    void SCR_CaptureConfig()
    {
        m_mGlobalSettings = new map<string, ref SCR_ConfigValue>();
        m_mModuleConfigs = new map<string, ref SCR_ModuleConfig>();
        m_bLoaded = false;
        m_sConfigPath = "$profile:CaptureConfig.txt";
        m_fLastModifiedTime = 0;

        // Set defaults
        SetDefaults();
    }

    //------------------------------------------------------------------------
    protected void SetDefaults()
    {
        // Session defaults
        SetString(SCR_ConfigKeys.SESSION_BASE_PATH, DEFAULT_BASE_PATH);
        SetString(SCR_ConfigKeys.SESSION_ID_FORMAT, "session_%t");
        SetBool(SCR_ConfigKeys.SESSION_AUTO_FINALIZE, true);

        // Capture defaults
        SetFloat(SCR_ConfigKeys.CAPTURE_INTERVAL_MS, DEFAULT_CAPTURE_INTERVAL_MS);
        SetFloat(SCR_ConfigKeys.CAPTURE_RATE_HZ, 1000.0 / DEFAULT_CAPTURE_INTERVAL_MS);
        SetInt(SCR_ConfigKeys.CAPTURE_MAX_FRAMES, 0);  // 0 = unlimited
        SetFloat(SCR_ConfigKeys.CAPTURE_MAX_DURATION_MS, 0);  // 0 = unlimited

        // Output defaults
        SetInt(SCR_ConfigKeys.OUTPUT_FORMAT, SCR_CaptureFormat.FORMAT_CSV);
        SetBool(SCR_ConfigKeys.OUTPUT_COMPRESSION, false);
        SetInt(SCR_ConfigKeys.OUTPUT_SPLIT_SIZE_MB, 0);  // 0 = no split

        // Buffer defaults
        SetInt(SCR_ConfigKeys.BUFFER_CAPACITY, DEFAULT_BUFFER_CAPACITY);
        SetInt(SCR_ConfigKeys.BUFFER_FLUSH_THRESHOLD, 256);
        SetFloat(SCR_ConfigKeys.BUFFER_FLUSH_INTERVAL_MS, 5000);
        SetInt(SCR_ConfigKeys.BUFFER_OVERFLOW_POLICY, SCR_BufferOverflowPolicy.OVERFLOW_DROP_OLDEST);

        // Logging defaults
        SetInt(SCR_ConfigKeys.LOG_LEVEL, LogLevel.NORMAL);
        SetInt(SCR_ConfigKeys.LOG_PROGRESS_INTERVAL, DEFAULT_LOG_PROGRESS_INTERVAL);
        SetBool(SCR_ConfigKeys.LOG_VERBOSE, false);

        // Module defaults
        SetString(SCR_ConfigKeys.MODULES_ENABLED, "*");  // All enabled by default
        SetString(SCR_ConfigKeys.MODULES_DISABLED, "");
    }

    //------------------------------------------------------------------------
    // Global setting accessors
    void SetValue(string key, SCR_ConfigValue value)
    {
        m_mGlobalSettings.Set(key, value);
    }

    SCR_ConfigValue GetValue(string key)
    {
        if (m_mGlobalSettings.Contains(key))
            return m_mGlobalSettings.Get(key);
        return null;
    }

    bool HasValue(string key)
    {
        return m_mGlobalSettings.Contains(key);
    }

    //------------------------------------------------------------------------
    // Typed setters
    void SetString(string key, string value)
    {
        SetValue(key, SCR_ConfigValue.FromString(value));
    }

    void SetFloat(string key, float value)
    {
        SetValue(key, SCR_ConfigValue.FromFloat(value));
    }

    void SetInt(string key, int value)
    {
        SetValue(key, SCR_ConfigValue.FromInt(value));
    }

    void SetBool(string key, bool value)
    {
        SetValue(key, SCR_ConfigValue.FromBool(value));
    }

    //------------------------------------------------------------------------
    // Typed getters with defaults
    string GetString(string key, string defaultValue = "")
    {
        SCR_ConfigValue cv = GetValue(key);
        if (cv)
            return cv.AsString();
        return defaultValue;
    }

    float GetFloat(string key, float defaultValue = 0)
    {
        SCR_ConfigValue cv = GetValue(key);
        if (cv)
            return cv.AsFloat();
        return defaultValue;
    }

    int GetInt(string key, int defaultValue = 0)
    {
        SCR_ConfigValue cv = GetValue(key);
        if (cv)
            return cv.AsInt();
        return defaultValue;
    }

    bool GetBool(string key, bool defaultValue = false)
    {
        SCR_ConfigValue cv = GetValue(key);
        if (cv)
            return cv.AsBool();
        return defaultValue;
    }

    //------------------------------------------------------------------------
    // Module configuration
    SCR_ModuleConfig GetModuleConfig(string moduleId)
    {
        if (!m_mModuleConfigs.Contains(moduleId))
        {
            SCR_ModuleConfig mc = new SCR_ModuleConfig(moduleId);
            // Inherit global defaults
            mc.SetIntervalMs(GetFloat(SCR_ConfigKeys.CAPTURE_INTERVAL_MS, DEFAULT_CAPTURE_INTERVAL_MS));
            m_mModuleConfigs.Set(moduleId, mc);
        }
        return m_mModuleConfigs.Get(moduleId);
    }

    void SetModuleConfig(string moduleId, SCR_ModuleConfig config)
    {
        m_mModuleConfigs.Set(moduleId, config);
    }

    bool IsModuleEnabled(string moduleId)
    {
        // Check explicit disable list
        string disabled = GetString(SCR_ConfigKeys.MODULES_DISABLED, "");
        if (disabled.Contains(moduleId))
            return false;

        // Check module-specific setting
        if (m_mModuleConfigs.Contains(moduleId))
            return m_mModuleConfigs.Get(moduleId).IsEnabled();

        // Check enabled list
        string enabled = GetString(SCR_ConfigKeys.MODULES_ENABLED, "*");
        if (enabled == "*")
            return true;

        return enabled.Contains(moduleId);
    }

    void SetModuleEnabled(string moduleId, bool enabled)
    {
        GetModuleConfig(moduleId).SetEnabled(enabled);
    }

    float GetModuleInterval(string moduleId)
    {
        if (m_mModuleConfigs.Contains(moduleId))
            return m_mModuleConfigs.Get(moduleId).GetIntervalMs();
        return GetFloat(SCR_ConfigKeys.CAPTURE_INTERVAL_MS, DEFAULT_CAPTURE_INTERVAL_MS);
    }

    void SetModuleInterval(string moduleId, float intervalMs)
    {
        GetModuleConfig(moduleId).SetIntervalMs(intervalMs);
    }

    //------------------------------------------------------------------------
    // Common computed properties
    string GetSessionBasePath()
    {
        return GetString(SCR_ConfigKeys.SESSION_BASE_PATH, DEFAULT_BASE_PATH);
    }

    float GetCaptureIntervalMs()
    {
        return GetFloat(SCR_ConfigKeys.CAPTURE_INTERVAL_MS, DEFAULT_CAPTURE_INTERVAL_MS);
    }

    float GetCaptureRateHz()
    {
        float interval = GetCaptureIntervalMs();
        if (interval <= 0)
            return 0;
        return 1000.0 / interval;
    }

    int GetBufferCapacity()
    {
        return GetInt(SCR_ConfigKeys.BUFFER_CAPACITY, DEFAULT_BUFFER_CAPACITY);
    }

    int GetOutputFormats()
    {
        return GetInt(SCR_ConfigKeys.OUTPUT_FORMAT, SCR_CaptureFormat.FORMAT_CSV);
    }

    bool IsVerbose()
    {
        return GetBool(SCR_ConfigKeys.LOG_VERBOSE, false);
    }

    //------------------------------------------------------------------------
    // Load configuration from file
    bool LoadFromFile(string path = "")
    {
        if (path.IsEmpty())
            path = m_sConfigPath;

        if (!FileIO.FileExists(path))
        {
            Print("[CaptureConfig] Config file not found: " + path, LogLevel.WARNING);
            return false;
        }

        FileHandle file = FileIO.OpenFile(path, FileMode.READ);
        if (!file)
        {
            Print("[CaptureConfig] Failed to open config file: " + path, LogLevel.ERROR);
            return false;
        }

        string line;
        string currentSection = "";

        while (file.ReadLine(line) >= 0)
        {
            line = line.Trim();

            // Skip empty lines and comments
            if (line.IsEmpty() || line.StartsWith("#") || line.StartsWith("//"))
                continue;

            // Section header
            if (line.StartsWith("[") && line.EndsWith("]"))
            {
                currentSection = line.Substring(1, line.Length() - 2);
                continue;
            }

            // Key=Value pair
            int eqPos = line.IndexOf("=");
            if (eqPos > 0)
            {
                string key = line.Substring(0, eqPos).Trim();
                string value = line.Substring(eqPos + 1, line.Length() - eqPos - 1).Trim();

                // Prepend section if present
                if (!currentSection.IsEmpty())
                    key = currentSection + "." + key;

                // Parse value type
                if (value == "true" || value == "false")
                    SetBool(key, value == "true");
                else if (value.Contains("."))
                    SetFloat(key, value.ToFloat());
                else if (IsNumeric(value))
                    SetInt(key, value.ToInt());
                else
                    SetString(key, value);
            }
        }

        file.Close();
        m_bLoaded = true;
        m_sConfigPath = path;

        Print("[CaptureConfig] Loaded configuration from: " + path, LogLevel.NORMAL);
        return true;
    }

    //------------------------------------------------------------------------
    // Save configuration to file
    bool SaveToFile(string path = "")
    {
        if (path.IsEmpty())
            path = m_sConfigPath;

        FileHandle file = FileIO.OpenFile(path, FileMode.WRITE);
        if (!file)
        {
            Print("[CaptureConfig] Failed to create config file: " + path, LogLevel.ERROR);
            return false;
        }

        // Write header
        file.WriteLine("# DriveDiT Capture Configuration");
        file.WriteLine("# Generated at runtime");
        file.WriteLine("");

        // Group settings by section
        ref map<string, ref array<string>> sections = new map<string, ref array<string>>();

        foreach (string key, SCR_ConfigValue value : m_mGlobalSettings)
        {
            string section = "global";
            string settingKey = key;

            int dotPos = key.IndexOf(".");
            if (dotPos > 0)
            {
                section = key.Substring(0, dotPos);
                settingKey = key.Substring(dotPos + 1, key.Length() - dotPos - 1);
            }

            if (!sections.Contains(section))
                sections.Set(section, new array<string>());

            sections.Get(section).Insert(settingKey + "=" + value.AsString());
        }

        // Write sections
        foreach (string section, array<string> settings : sections)
        {
            file.WriteLine("[" + section + "]");
            foreach (string setting : settings)
            {
                file.WriteLine(setting);
            }
            file.WriteLine("");
        }

        // Write module configs
        file.WriteLine("[modules]");
        foreach (string moduleId, SCR_ModuleConfig mc : m_mModuleConfigs)
        {
            string enabledStr = "false";
            if (mc.IsEnabled())
                enabledStr = "true";
            file.WriteLine(moduleId + ".enabled=" + enabledStr);
            file.WriteLine(moduleId + ".interval_ms=" + mc.GetIntervalMs().ToString());
            file.WriteLine(moduleId + ".priority=" + mc.GetPriority().ToString());
        }

        file.Close();
        Print("[CaptureConfig] Saved configuration to: " + path, LogLevel.NORMAL);
        return true;
    }

    //------------------------------------------------------------------------
    // Helper to check if string is numeric
    protected bool IsNumeric(string str)
    {
        if (str.IsEmpty())
            return false;

        for (int i = 0; i < str.Length(); i++)
        {
            string ch = str.Substring(i, 1);
            if (ch != "-" && ch != "0" && ch != "1" && ch != "2" && ch != "3" &&
                ch != "4" && ch != "5" && ch != "6" && ch != "7" && ch != "8" && ch != "9")
                return false;
        }
        return true;
    }

    //------------------------------------------------------------------------
    // Create preset configurations
    static SCR_CaptureConfig CreateMinimalConfig()
    {
        SCR_CaptureConfig config = new SCR_CaptureConfig();
        config.SetFloat(SCR_ConfigKeys.CAPTURE_INTERVAL_MS, 500);  // 2 Hz
        config.SetInt(SCR_ConfigKeys.BUFFER_CAPACITY, 256);
        config.SetBool(SCR_ConfigKeys.LOG_VERBOSE, false);
        return config;
    }

    static SCR_CaptureConfig CreateHighFrequencyConfig()
    {
        SCR_CaptureConfig config = new SCR_CaptureConfig();
        config.SetFloat(SCR_ConfigKeys.CAPTURE_INTERVAL_MS, 50);   // 20 Hz
        config.SetInt(SCR_ConfigKeys.BUFFER_CAPACITY, 4096);
        config.SetInt(SCR_ConfigKeys.BUFFER_FLUSH_THRESHOLD, 1024);
        config.SetFloat(SCR_ConfigKeys.BUFFER_FLUSH_INTERVAL_MS, 2000);
        return config;
    }

    static SCR_CaptureConfig CreateMLTrainingConfig()
    {
        SCR_CaptureConfig config = new SCR_CaptureConfig();
        config.SetFloat(SCR_ConfigKeys.CAPTURE_INTERVAL_MS, 200);  // 5 Hz (comma.ai standard)
        config.SetInt(SCR_ConfigKeys.OUTPUT_FORMAT, SCR_CaptureFormat.FORMAT_CSV | SCR_CaptureFormat.FORMAT_BINARY);
        config.SetInt(SCR_ConfigKeys.BUFFER_CAPACITY, 2048);
        config.SetBool(SCR_ConfigKeys.LOG_VERBOSE, true);
        config.SetInt(SCR_ConfigKeys.LOG_PROGRESS_INTERVAL, 250);
        return config;
    }

    //------------------------------------------------------------------------
    // Debug output
    string GetDebugString()
    {
        string str = "[CaptureConfig]\n";
        str += "  Loaded: " + m_bLoaded.ToString() + "\n";
        str += "  Path: " + m_sConfigPath + "\n";
        str += "  Global settings: " + m_mGlobalSettings.Count().ToString() + "\n";
        str += "  Module configs: " + m_mModuleConfigs.Count().ToString() + "\n";
        str += "  Capture rate: " + GetCaptureRateHz().ToString(5, 1) + " Hz\n";
        str += "  Buffer capacity: " + GetBufferCapacity().ToString() + "\n";
        return str;
    }
}
