// ============================================================================
// SCR_CaptureSerializer - Multi-Format Data Serialization
// ============================================================================
//
// Handles serialization of captured data to multiple output formats:
// - CSV (default, human-readable, ML pipeline compatible)
// - Binary (compact, fast writes, schema versioned)
// - JSON (optional, for debugging/interchange)
//
// DESIGN PRINCIPLES:
// - Format-agnostic interface for writers
// - Schema versioning for binary compatibility
// - Batched writes for I/O efficiency
// - Automatic file splitting for large datasets
//
// ============================================================================

// -----------------------------------------------------------------------------
// Serializer state
// -----------------------------------------------------------------------------
enum SCR_SerializerState
{
    SERIALIZER_UNINITIALIZED,
    SERIALIZER_READY,
    SERIALIZER_WRITING,
    SERIALIZER_ERROR,
    SERIALIZER_FINALIZED
}

// -----------------------------------------------------------------------------
// SCR_SerializerStats - Write statistics
// -----------------------------------------------------------------------------
class SCR_SerializerStats
{
    int m_iTotalRecords;
    int m_iTotalBytes;
    int m_iFileCount;
    int m_iErrorCount;
    float m_fTotalWriteTimeMs;
    float m_fLastWriteTimeMs;

    //------------------------------------------------------------------------
    void SCR_SerializerStats()
    {
        Reset();
    }

    void Reset()
    {
        m_iTotalRecords = 0;
        m_iTotalBytes = 0;
        m_iFileCount = 0;
        m_iErrorCount = 0;
        m_fTotalWriteTimeMs = 0;
        m_fLastWriteTimeMs = 0;
    }

    float GetAverageWriteTimeMs()
    {
        if (m_iTotalRecords == 0)
            return 0;
        return m_fTotalWriteTimeMs / m_iTotalRecords;
    }

    float GetThroughputRecordsPerSec()
    {
        if (m_fTotalWriteTimeMs == 0)
            return 0;
        return (m_iTotalRecords * 1000.0) / m_fTotalWriteTimeMs;
    }
}

// -----------------------------------------------------------------------------
// SCR_IFormatWriter - Interface for format-specific writers
// -----------------------------------------------------------------------------
class SCR_IFormatWriter
{
    protected SCR_SerializerState m_eState;
    protected string m_sFilePath;
    protected ref FileHandle m_File;
    protected int m_iBytesWritten;
    protected int m_iRecordsWritten;

    //------------------------------------------------------------------------
    void SCR_IFormatWriter()
    {
        m_eState = SCR_SerializerState.SERIALIZER_UNINITIALIZED;
        m_sFilePath = "";
        m_iBytesWritten = 0;
        m_iRecordsWritten = 0;
    }

    //------------------------------------------------------------------------
    // Open file for writing
    bool Open(string filePath)
    {
        m_sFilePath = filePath;
        m_File = FileIO.OpenFile(filePath, FileMode.WRITE);
        if (!m_File)
        {
            m_eState = SCR_SerializerState.SERIALIZER_ERROR;
            return false;
        }
        m_eState = SCR_SerializerState.SERIALIZER_READY;
        return true;
    }

    //------------------------------------------------------------------------
    // Write header (format-specific)
    bool WriteHeader(array<string> columns) { return true; }

    //------------------------------------------------------------------------
    // Write a data record
    bool WriteRecord(SCR_CaptureDataRecord record) { return false; }

    //------------------------------------------------------------------------
    // Write raw string line (for CSV)
    bool WriteLine(string line) { return false; }

    //------------------------------------------------------------------------
    // Flush pending writes
    void Flush() { }

    //------------------------------------------------------------------------
    // Close file
    void Close()
    {
        if (m_File)
        {
            m_File.Close();
            m_File = null;
        }
        m_eState = SCR_SerializerState.SERIALIZER_FINALIZED;
    }

    //------------------------------------------------------------------------
    // Accessors
    SCR_SerializerState GetState() { return m_eState; }
    string GetFilePath() { return m_sFilePath; }
    int GetBytesWritten() { return m_iBytesWritten; }
    int GetRecordsWritten() { return m_iRecordsWritten; }
    bool IsOpen() { return m_File != null && m_eState == SCR_SerializerState.SERIALIZER_READY; }
}

// -----------------------------------------------------------------------------
// SCR_CSVWriter - CSV format writer
// -----------------------------------------------------------------------------
class SCR_CSVWriter : SCR_IFormatWriter
{
    protected string m_sDelimiter;
    protected bool m_bHeaderWritten;
    protected ref array<string> m_aColumnOrder;

    //------------------------------------------------------------------------
    void SCR_CSVWriter()
    {
        m_sDelimiter = ",";
        m_bHeaderWritten = false;
        m_aColumnOrder = new array<string>();
    }

    //------------------------------------------------------------------------
    void SetDelimiter(string delimiter)
    {
        m_sDelimiter = delimiter;
    }

    //------------------------------------------------------------------------
    override bool WriteHeader(array<string> columns)
    {
        if (!m_File || m_bHeaderWritten)
            return false;

        if (!columns || columns.IsEmpty())
            return false;

        m_aColumnOrder.Clear();
        string headerLine = "";

        for (int i = 0; i < columns.Count(); i++)
        {
            if (i > 0)
                headerLine += m_sDelimiter;
            headerLine += columns[i];
            m_aColumnOrder.Insert(columns[i]);
        }

        m_File.WriteLine(headerLine);
        m_bHeaderWritten = true;
        m_iBytesWritten += headerLine.Length() + 2;  // +2 for CRLF

        return true;
    }

    //------------------------------------------------------------------------
    override bool WriteRecord(SCR_CaptureDataRecord record)
    {
        if (!m_File || !record)
            return false;

        string csvLine = record.ToCSV();
        if (csvLine.IsEmpty())
            return false;

        m_File.WriteLine(csvLine);
        m_iRecordsWritten++;
        m_iBytesWritten += csvLine.Length() + 2;

        return true;
    }

    //------------------------------------------------------------------------
    override bool WriteLine(string line)
    {
        if (!m_File)
            return false;

        m_File.WriteLine(line);
        m_iRecordsWritten++;
        m_iBytesWritten += line.Length() + 2;

        return true;
    }

    //------------------------------------------------------------------------
    bool IsHeaderWritten()
    {
        return m_bHeaderWritten;
    }
}

// -----------------------------------------------------------------------------
// SCR_BinaryWriter - Binary format writer with schema versioning
// -----------------------------------------------------------------------------
class SCR_BinaryWriter : SCR_IFormatWriter
{
    protected int m_iSchemaVersion;
    protected bool m_bHeaderWritten;

    // Binary file header magic bytes
    static const int MAGIC_HEADER = 0x44524956;  // "DRIV" in ASCII

    //------------------------------------------------------------------------
    void SCR_BinaryWriter()
    {
        m_iSchemaVersion = 1;
        m_bHeaderWritten = false;
    }

    //------------------------------------------------------------------------
    void SetSchemaVersion(int version)
    {
        m_iSchemaVersion = version;
    }

    //------------------------------------------------------------------------
    override bool Open(string filePath)
    {
        if (!super.Open(filePath))
            return false;

        // Write binary header
        WriteBinaryHeader();
        return true;
    }

    //------------------------------------------------------------------------
    protected void WriteBinaryHeader()
    {
        if (!m_File)
            return;

        // Magic number (4 bytes)
        m_File.Write(MAGIC_HEADER, 4);

        // Schema version (4 bytes)
        m_File.Write(m_iSchemaVersion, 4);

        // Timestamp (8 bytes - placeholder)
        float timestamp = GetGame().GetWorld().GetWorldTime();
        m_File.Write(timestamp, 4);
        m_File.Write(0, 4);  // Reserved

        // Reserved header space (16 bytes)
        for (int i = 0; i < 4; i++)
        {
            m_File.Write(0, 4);
        }

        m_iBytesWritten += 32;
        m_bHeaderWritten = true;
    }

    //------------------------------------------------------------------------
    override bool WriteRecord(SCR_CaptureDataRecord record)
    {
        if (!m_File || !record)
            return false;

        record.ToBinary(m_File);
        m_iRecordsWritten++;

        return true;
    }

    //------------------------------------------------------------------------
    // Write raw bytes
    bool WriteBytes(int data, int length)
    {
        if (!m_File)
            return false;

        m_File.Write(data, length);
        m_iBytesWritten += length;
        return true;
    }

    bool WriteFloat(float data)
    {
        if (!m_File)
            return false;

        m_File.Write(data, 4);
        m_iBytesWritten += 4;
        return true;
    }

    int GetSchemaVersion()
    {
        return m_iSchemaVersion;
    }
}

// -----------------------------------------------------------------------------
// SCR_CaptureSerializer - Main serializer class
// -----------------------------------------------------------------------------
class SCR_CaptureSerializer
{
    // Configuration
    protected ref SCR_CaptureConfig m_Config;
    protected string m_sSessionPath;
    protected int m_iActiveFormats;

    // Writers
    protected ref map<string, ref SCR_CSVWriter> m_mCSVWriters;
    protected ref map<string, ref SCR_BinaryWriter> m_mBinaryWriters;

    // State
    protected SCR_SerializerState m_eState;
    protected ref SCR_SerializerStats m_Stats;

    // File management
    protected int m_iCurrentFileIndex;
    protected int m_iMaxFileSizeBytes;
    protected bool m_bAutoSplit;

    //------------------------------------------------------------------------
    void SCR_CaptureSerializer()
    {
        m_mCSVWriters = new map<string, ref SCR_CSVWriter>();
        m_mBinaryWriters = new map<string, ref SCR_BinaryWriter>();
        m_eState = SCR_SerializerState.SERIALIZER_UNINITIALIZED;
        m_Stats = new SCR_SerializerStats();
        m_iCurrentFileIndex = 0;
        m_iMaxFileSizeBytes = 0;
        m_bAutoSplit = false;
        m_iActiveFormats = SCR_CaptureFormat.FORMAT_CSV;
    }

    //------------------------------------------------------------------------
    // Initialize with configuration
    SCR_CaptureResult Initialize(SCR_CaptureConfig config, string sessionPath)
    {
        if (!config)
            return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_CONFIG_INVALID, "Config is null");

        m_Config = config;
        m_sSessionPath = sessionPath;
        m_iActiveFormats = config.GetOutputFormats();

        // Create session directory
        if (!FileIO.MakeDirectory(sessionPath))
        {
            Print("[CaptureSerializer] Warning: Could not create directory: " + sessionPath, LogLevel.WARNING);
        }

        // Configure file splitting
        int splitSizeMB = config.GetInt(SCR_ConfigKeys.OUTPUT_SPLIT_SIZE_MB, 0);
        if (splitSizeMB > 0)
        {
            m_iMaxFileSizeBytes = splitSizeMB * 1024 * 1024;
            m_bAutoSplit = true;
        }

        m_eState = SCR_SerializerState.SERIALIZER_READY;
        Print("[CaptureSerializer] Initialized at: " + sessionPath, LogLevel.NORMAL);

        return SCR_CaptureResult.Success();
    }

    //------------------------------------------------------------------------
    // Get or create CSV writer for a module
    SCR_CSVWriter GetCSVWriter(string moduleId, string header = "")
    {
        if (!(m_iActiveFormats & SCR_CaptureFormat.FORMAT_CSV))
            return null;

        string key = moduleId;
        if (!m_mCSVWriters.Contains(key))
        {
            // Create new writer
            SCR_CSVWriter writer = new SCR_CSVWriter();
            string filePath = BuildFilePath(moduleId, "csv");

            if (!writer.Open(filePath))
            {
                Print("[CaptureSerializer] Failed to create CSV file: " + filePath, LogLevel.ERROR);
                return null;
            }

            // Write header if provided
            if (!header.IsEmpty())
            {
                ref array<string> columns = new array<string>();
                header.Split(",", columns, false);
                writer.WriteHeader(columns);
            }

            m_mCSVWriters.Set(key, writer);
            m_Stats.m_iFileCount++;

            Print("[CaptureSerializer] Created CSV file: " + filePath, LogLevel.NORMAL);
        }

        return m_mCSVWriters.Get(key);
    }

    //------------------------------------------------------------------------
    // Get or create Binary writer for a module
    SCR_BinaryWriter GetBinaryWriter(string moduleId, int schemaVersion = 1)
    {
        if (!(m_iActiveFormats & SCR_CaptureFormat.FORMAT_BINARY))
            return null;

        string key = moduleId;
        if (!m_mBinaryWriters.Contains(key))
        {
            SCR_BinaryWriter writer = new SCR_BinaryWriter();
            writer.SetSchemaVersion(schemaVersion);

            string filePath = BuildFilePath(moduleId, "bin");

            if (!writer.Open(filePath))
            {
                Print("[CaptureSerializer] Failed to create binary file: " + filePath, LogLevel.ERROR);
                return null;
            }

            m_mBinaryWriters.Set(key, writer);
            m_Stats.m_iFileCount++;

            Print("[CaptureSerializer] Created binary file: " + filePath, LogLevel.NORMAL);
        }

        return m_mBinaryWriters.Get(key);
    }

    //------------------------------------------------------------------------
    // Write a record (dispatches to appropriate writers)
    SCR_CaptureResult WriteRecord(SCR_CaptureDataRecord record)
    {
        if (!record)
            return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_SERIALIZATION, "Null record");

        if (m_eState != SCR_SerializerState.SERIALIZER_READY)
            return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_NOT_INITIALIZED, "Serializer not ready");

        float startTimeMs = GetGame().GetWorld().GetWorldTime();
        string moduleId = record.GetModuleId();
        int bytesWritten = 0;
        bool anySuccess = false;

        // Write to CSV if enabled
        if (m_iActiveFormats & SCR_CaptureFormat.FORMAT_CSV)
        {
            SCR_CSVWriter csvWriter = GetCSVWriter(moduleId);
            if (csvWriter && csvWriter.WriteRecord(record))
            {
                bytesWritten += csvWriter.GetBytesWritten();
                anySuccess = true;
            }
        }

        // Write to Binary if enabled
        if (m_iActiveFormats & SCR_CaptureFormat.FORMAT_BINARY)
        {
            SCR_BinaryWriter binWriter = GetBinaryWriter(moduleId);
            if (binWriter && binWriter.WriteRecord(record))
            {
                bytesWritten += binWriter.GetBytesWritten();
                anySuccess = true;
            }
        }

        // Update stats
        float writeTimeMs = GetGame().GetWorld().GetWorldTime() - startTimeMs;
        m_Stats.m_fTotalWriteTimeMs += writeTimeMs;
        m_Stats.m_fLastWriteTimeMs = writeTimeMs;

        if (anySuccess)
        {
            m_Stats.m_iTotalRecords++;
            m_Stats.m_iTotalBytes += bytesWritten;
            return SCR_CaptureResult.Success(bytesWritten, writeTimeMs);
        }

        m_Stats.m_iErrorCount++;
        return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_IO_FAILURE, "No writers succeeded");
    }

    //------------------------------------------------------------------------
    // Write raw CSV line for a module
    SCR_CaptureResult WriteCSVLine(string moduleId, string line, string header = "")
    {
        if (m_eState != SCR_SerializerState.SERIALIZER_READY)
            return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_NOT_INITIALIZED, "Serializer not ready");

        SCR_CSVWriter writer = GetCSVWriter(moduleId, header);
        if (!writer)
            return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_IO_FAILURE, "No CSV writer available");

        if (writer.WriteLine(line))
        {
            m_Stats.m_iTotalRecords++;
            m_Stats.m_iTotalBytes += line.Length() + 2;
            return SCR_CaptureResult.Success(line.Length() + 2);
        }

        m_Stats.m_iErrorCount++;
        return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_IO_FAILURE, "Write failed");
    }

    //------------------------------------------------------------------------
    // Build file path for a module
    protected string BuildFilePath(string moduleId, string extension)
    {
        string fileName = moduleId;

        if (m_bAutoSplit && m_iCurrentFileIndex > 0)
            fileName += "_" + m_iCurrentFileIndex.ToString();

        fileName += "." + extension;

        return m_sSessionPath + "/" + fileName;
    }

    //------------------------------------------------------------------------
    // Flush all writers
    void FlushAll()
    {
        foreach (string key, SCR_CSVWriter writer : m_mCSVWriters)
        {
            if (writer)
                writer.Flush();
        }

        foreach (string key, SCR_BinaryWriter writer : m_mBinaryWriters)
        {
            if (writer)
                writer.Flush();
        }
    }

    //------------------------------------------------------------------------
    // Finalize and close all writers
    SCR_CaptureResult Finalize()
    {
        // Close CSV writers
        foreach (string key, SCR_CSVWriter writer : m_mCSVWriters)
        {
            if (writer)
                writer.Close();
        }
        m_mCSVWriters.Clear();

        // Close Binary writers
        foreach (string key, SCR_BinaryWriter writer : m_mBinaryWriters)
        {
            if (writer)
                writer.Close();
        }
        m_mBinaryWriters.Clear();

        m_eState = SCR_SerializerState.SERIALIZER_FINALIZED;

        Print("[CaptureSerializer] Finalized. Total records: " + m_Stats.m_iTotalRecords.ToString() +
              ", Total bytes: " + m_Stats.m_iTotalBytes.ToString(), LogLevel.NORMAL);

        return SCR_CaptureResult.Success(m_Stats.m_iTotalBytes);
    }

    //------------------------------------------------------------------------
    // Write session metadata
    void WriteSessionMetadata(string key, string value)
    {
        string metadataPath = m_sSessionPath + "/metadata.txt";
        FileHandle file = FileIO.OpenFile(metadataPath, FileMode.APPEND);
        if (file)
        {
            file.WriteLine(key + "=" + value);
            file.Close();
        }
    }

    //------------------------------------------------------------------------
    // Write session summary
    void WriteSessionSummary()
    {
        string summaryPath = m_sSessionPath + "/summary.txt";
        FileHandle file = FileIO.OpenFile(summaryPath, FileMode.WRITE);
        if (!file)
            return;

        file.WriteLine("=== CAPTURE SESSION SUMMARY ===");
        file.WriteLine("total_records=" + m_Stats.m_iTotalRecords.ToString());
        file.WriteLine("total_bytes=" + m_Stats.m_iTotalBytes.ToString());
        file.WriteLine("file_count=" + m_Stats.m_iFileCount.ToString());
        file.WriteLine("error_count=" + m_Stats.m_iErrorCount.ToString());
        file.WriteLine("avg_write_time_ms=" + m_Stats.GetAverageWriteTimeMs().ToString(8, 3));
        file.WriteLine("throughput_records_per_sec=" + m_Stats.GetThroughputRecordsPerSec().ToString(8, 1));
        file.WriteLine("");

        file.WriteLine("=== FILES CREATED ===");
        foreach (string key, SCR_CSVWriter writer : m_mCSVWriters)
        {
            if (writer)
            {
                file.WriteLine("csv: " + writer.GetFilePath() +
                              " (" + writer.GetRecordsWritten().ToString() + " records, " +
                              writer.GetBytesWritten().ToString() + " bytes)");
            }
        }
        foreach (string key, SCR_BinaryWriter writer : m_mBinaryWriters)
        {
            if (writer)
            {
                file.WriteLine("bin: " + writer.GetFilePath() +
                              " (" + writer.GetRecordsWritten().ToString() + " records)");
            }
        }

        file.Close();
    }

    //------------------------------------------------------------------------
    // Accessors
    SCR_SerializerState GetState() { return m_eState; }
    string GetSessionPath() { return m_sSessionPath; }
    SCR_SerializerStats GetStats() { return m_Stats; }
    int GetActiveFormats() { return m_iActiveFormats; }

    bool HasFormat(SCR_CaptureFormat format)
    {
        return (m_iActiveFormats & format) != 0;
    }

    //------------------------------------------------------------------------
    // Debug output
    string GetDebugString()
    {
        string str = "[Serializer] state=" + m_eState.ToString();
        str += " path=" + m_sSessionPath;
        str += " records=" + m_Stats.m_iTotalRecords.ToString();
        str += " bytes=" + m_Stats.m_iTotalBytes.ToString();
        str += " files=" + m_Stats.m_iFileCount.ToString();
        return str;
    }
}
