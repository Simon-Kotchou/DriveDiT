// ============================================================================
// DC_ProbeCsv - acknowledged CSV and key=value writer of the T0-ENG probes (DD-4-probe)
// ============================================================================
//
// Every probe file goes through this writer. One line = one FileHandle.Write(line + "\n") call, and the line counts
// only when Write returns exactly line.Length() + 1 bytes (FileHandle.c:98 documents "the total number of bytes
// successfully written"). The first short or failed write latches HasFailed() and refuses every later write, so a
// file never continues after a gap. WriteLine is never used for data: it appends CR LF (FileHandle.c:103).
//
// The writer is also a measuring instrument: probe E-19 (DC_ProbeEngine) tests whether the Write return value can be
// trusted at all. Every failure message therefore carries the requested and the returned byte counts.
//
// CSV rules (PROBE_DESIGN.md 4.1): ASCII, ',' separator, LF line ends, one header row, no quoting. Callers pass only
// field text without ',', '"', CR or LF (CleanField makes any text safe). Booleans are 1/0, missing values are empty.
// Matrices are 12 columns in the dc-2.0 hf order pos, right, up, fwd (AddMat34Columns / DC_ProbeRow.AddMat34).
// ============================================================================

//------------------------------------------------------------------------------------------------
//! One CSV row under construction. Values are appended in column order.
class DC_ProbeRow
{
	protected ref array<string> m_aValues = {};

	//------------------------------------------------------------------------------------------------
	void Clear()
	{
		m_aValues.Clear();
	}

	//------------------------------------------------------------------------------------------------
	//! Caller guarantees ASCII without ',', '"', CR, LF (see DC_ProbeCsv.CleanField)
	void AddString(string value)
	{
		m_aValues.Insert(value);
	}

	//------------------------------------------------------------------------------------------------
	void AddInt(int value)
	{
		m_aValues.Insert(value.ToString());
	}

	//------------------------------------------------------------------------------------------------
	//! value.ToString(-1, decimals)
	void AddFloat(float value, int decimals)
	{
		m_aValues.Insert(value.ToString(-1, decimals));
	}

	//------------------------------------------------------------------------------------------------
	//! "1" / "0"
	void AddBool(bool value)
	{
		m_aValues.Insert(DC_LaunchGuard.BoolText(value));
	}

	//------------------------------------------------------------------------------------------------
	void AddEmpty()
	{
		m_aValues.Insert(string.Empty);
	}

	//------------------------------------------------------------------------------------------------
	//! 3 fields: x, y, z
	void AddVector(vector value, int decimals)
	{
		AddFloat(value[0], decimals);
		AddFloat(value[1], decimals);
		AddFloat(value[2], decimals);
	}

	//------------------------------------------------------------------------------------------------
	//! 12 fields in dc-2.0 hf order: pos x,y,z (posDecimals), right x,y,z, up x,y,z, fwd x,y,z (axisDecimals).
	//! mat rows are right (0), up (1), forward (2), position (3).
	void AddMat34(vector mat[4], int axisDecimals, int posDecimals)
	{
		AddVector(mat[3], posDecimals);
		AddVector(mat[0], axisDecimals);
		AddVector(mat[1], axisDecimals);
		AddVector(mat[2], axisDecimals);
	}

	//------------------------------------------------------------------------------------------------
	int Count()
	{
		return m_aValues.Count();
	}

	//------------------------------------------------------------------------------------------------
	//! The values joined with ','. Empty fields are kept (joined by hand, so no skip rule can drop one).
	string Join()
	{
		string line;
		for (int i = 0, count = m_aValues.Count(); i < count; i++)
		{
			if (i > 0)
				line += ",";

			line += m_aValues[i];
		}

		return line;
	}
}

//------------------------------------------------------------------------------------------------
//! Acknowledged CSV writer: every line goes through FileHandle.Write(line + "\n") and counts only when Write
//! returns exactly line.Length() + 1 (E-19 decides whether that return can be trusted; the probe writer is also the
//! measuring instrument). The first failure latches HasFailed(); later writes are refused.
class DC_ProbeCsv
{
	protected ref FileHandle m_File;
	protected string m_sPath;
	protected int m_iColumnCount;
	protected int m_iRowCount;		//!< acknowledged data rows
	protected int m_iByteCount;		//!< acknowledged bytes, header included
	protected bool m_bOpened;		//!< Open was called (a writer is used for one file only)
	protected bool m_bFailed;
	protected string m_sFailure;

	//------------------------------------------------------------------------------------------------
	//! Creates path (refuses an existing file: FileIO.FileExists) and writes the header row. false on any failure.
	bool Open(string path, notnull array<string> columns)
	{
		if (m_bOpened)
			return Fail("Open called twice (" + m_sPath + ", then " + path + ")");

		m_bOpened = true;
		m_sPath = path;
		m_iColumnCount = columns.Count();
		if (m_iColumnCount == 0)
			return Fail("no columns for " + path);

		if (FileIO.FileExists(path))
			return Fail("refusing to overwrite the existing file " + path);

		m_File = FileIO.OpenFile(path, FileMode.WRITE);
		if (!m_File)
			return Fail("FileIO.OpenFile(" + path + ", WRITE) returned null");

		string header;
		for (int i = 0; i < m_iColumnCount; i++)
		{
			if (i > 0)
				header += ",";

			header += columns[i];
		}

		return WriteAcknowledged(header);
	}

	//------------------------------------------------------------------------------------------------
	//! Appends one data row; false (and HasFailed) on a short or failed write or when row.Count() != column count
	bool WriteRow(notnull DC_ProbeRow row)
	{
		if (m_bFailed)
			return false;

		if (!m_File)
			return Fail("WriteRow on a writer that is not open (" + m_sPath + ")");

		if (row.Count() != m_iColumnCount)
			return Fail(string.Format("row with %1 values for %2 columns in %3", row.Count(), m_iColumnCount, m_sPath));

		if (!WriteAcknowledged(row.Join()))
			return false;

		m_iRowCount++;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! One Write(line + "\n"); counts the bytes only when the return equals the length
	protected bool WriteAcknowledged(string line)
	{
		string data = line + "\n";
		int expected = data.Length();
		int written = m_File.Write(data);
		if (written != expected)
			return Fail(string.Format("short or failed write in %1: FileHandle.Write returned %2 of %3 bytes", m_sPath, written, expected));

		m_iByteCount += expected;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Latches the first failure. \return false
	protected bool Fail(string reason)
	{
		if (!m_bFailed)
		{
			m_bFailed = true;
			m_sFailure = reason;
		}

		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! Idempotent
	void Close()
	{
		if (!m_File)
			return;

		m_File.Close();
		m_File = null;
	}

	//------------------------------------------------------------------------------------------------
	bool IsOpen()
	{
		if (!m_File)
			return false;

		return m_File.IsOpen();
	}

	//------------------------------------------------------------------------------------------------
	bool HasFailed()
	{
		return m_bFailed;
	}

	//------------------------------------------------------------------------------------------------
	//! First failure, one line
	string GetFailure()
	{
		return m_sFailure;
	}

	//------------------------------------------------------------------------------------------------
	string GetPath()
	{
		return m_sPath;
	}

	//------------------------------------------------------------------------------------------------
	//! Acknowledged data rows (header excluded)
	int GetRowCount()
	{
		return m_iRowCount;
	}

	//------------------------------------------------------------------------------------------------
	//! Acknowledged bytes (header included)
	int GetByteCount()
	{
		return m_iByteCount;
	}

	//------------------------------------------------------------------------------------------------
	int GetColumnCount()
	{
		return m_iColumnCount;
	}

	//------------------------------------------------------------------------------------------------
	//! Appends <prefix>_pos_x .. <prefix>_fwd_z (12 names) in DC_ProbeRow.AddMat34 order
	static void AddMat34Columns(notnull array<string> columns, string prefix)
	{
		AddVectorColumns(columns, prefix + "_pos");
		AddVectorColumns(columns, prefix + "_right");
		AddVectorColumns(columns, prefix + "_up");
		AddVectorColumns(columns, prefix + "_fwd");
	}

	//------------------------------------------------------------------------------------------------
	//! Appends <prefix>_x, <prefix>_y, <prefix>_z
	static void AddVectorColumns(notnull array<string> columns, string prefix)
	{
		columns.Insert(prefix + "_x");
		columns.Insert(prefix + "_y");
		columns.Insert(prefix + "_z");
	}

	//------------------------------------------------------------------------------------------------
	//! Writes a whole text file (key=value lines) with the same acknowledgement; refuses an existing file
	static bool WriteTextFile(string path, notnull array<string> lines)
	{
		string failure;
		return WriteLines(path, lines, false, failure);
	}

	//------------------------------------------------------------------------------------------------
	//! Appends lines to an existing text file (FileMode.APPEND) with the same acknowledgement; refuses a missing file.
	//! DC_ProbeContext.AppendHeaderLines uses it for run-header lines a probe only knows after OnProbeStart.
	static bool AppendTextFile(string path, notnull array<string> lines)
	{
		string failure;
		return WriteLines(path, lines, true, failure);
	}

	//------------------------------------------------------------------------------------------------
	//! WriteTextFile / AppendTextFile with the failure text
	//! \param[out] failure one line, empty on success
	static bool WriteLines(string path, notnull array<string> lines, bool append, out string failure)
	{
		failure = string.Empty;
		bool exists = FileIO.FileExists(path);
		if (!append && exists)
		{
			failure = "refusing to overwrite the existing file " + path;
			return false;
		}

		if (append && !exists)
		{
			failure = "cannot append to the missing file " + path;
			return false;
		}

		FileHandle file;
		if (append)
			file = FileIO.OpenFile(path, FileMode.APPEND);
		else
			file = FileIO.OpenFile(path, FileMode.WRITE);

		if (!file)
		{
			failure = "FileIO.OpenFile(" + path + ") returned null";
			return false;
		}

		foreach (string line : lines)
		{
			string data = CleanLine(line) + "\n";
			int expected = data.Length();
			int written = file.Write(data);
			if (written != expected)
			{
				failure = string.Format("short or failed write in %1: FileHandle.Write returned %2 of %3 bytes", path, written, expected);
				file.Close();
				return false;
			}
		}

		file.Close();
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Makes any text safe as one CSV field: ',', '"', ';', CR, LF, TAB and spaces become '_'
	static string CleanField(string value)
	{
		string clean = value;
		clean.Replace(",", "_");
		clean.Replace("\"", "_");
		clean.Replace(";", "_");
		clean.Replace("\r", "_");
		clean.Replace("\n", "_");
		clean.Replace("\t", "_");
		clean.Replace(" ", "_");
		return clean;
	}

	//------------------------------------------------------------------------------------------------
	//! Makes any text safe as one key=value line: CR and LF become ' '
	static string CleanLine(string value)
	{
		string clean = value;
		clean.Replace("\r", " ");
		clean.Replace("\n", " ");
		return clean;
	}
}
