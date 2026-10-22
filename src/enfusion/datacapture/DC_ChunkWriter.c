// ============================================================================
// DC_ChunkWriter - acknowledged chunk writer of one dc-2.0 stream (DD-4-min)
// ============================================================================
//
// Writes <session>/<stream>/chunk_NNNNN.csv and <session>/<stream>/commit.csv by the rules of
// docs/capture_contract/dc-2.0.md section 8 (TBD-3 decides the semantics; E-19 of DD-4-probe measures them):
//   - rows are buffered per chunk (DC_ProbeRow, joined with ','); a chunk is closed at chunk_max_rows, after
//     clocks.chunk_max_s of world time since its first row (FlushIfDue, called by DC_Session every frame) or at the
//     end, and only when it holds at least one row;
//   - a chunk is written whole, header plus rows, by ONE FileHandle.Write into a new file (an existing file is
//     refused) and closed. The Write return must equal the byte count (ASCII: one byte per character); with
//     verifyLength (TBD-3 fallback) the closed file's length is also read back through a READ handle;
//   - only then is the commit line chunk_id,first_key,last_key,n_rows,n_bytes appended to commit.csv by one
//     open-APPEND-write-close cycle (E-19 c11), or through the handle kept open for the session (keepCommitOpen, the
//     e19kill e19_commit.csv pattern). commit.csv gets its header when Open creates it. A row is acknowledged only
//     when its chunk's commit line was written; with verifyLength commit.csv must also hold exactly the bytes written
//     so far (header and every line, read back through the handle kept open or a READ handle on the closed file)
//     before the rows of the line are counted, and a log kept open is read back once more after its close;
//   - the first short or failed write latches HasFailed(): nothing is written or acknowledged afterwards, the owner
//     reports it (DC_Session.FailWrite) and the session ends with script_error:write_fail. Every failure message
//     carries the requested and the returned byte counts. WriteLine is never used (it appends CR LF).
//
// A crash loses at most the chunk in memory and leaves a consistent commit log: a chunk file whose commit line is
// missing is an uncommitted tail that readers discard. No file is ever rewritten in place.
//
// DC_Session owns every writer (RegisterStream): it commits them at the seal, counts them in session_end.json and
// can still close them on world teardown. Abandon drops the buffered rows without writing them (after a write
// failure no further row is acknowledged).
// ============================================================================

//------------------------------------------------------------------------------------------------
class DC_ChunkWriter : DC_SessionStream
{
	static const string COMMIT_NAME = "commit.csv";
	static const string COMMIT_HEADER = "chunk_id,first_key,last_key,n_rows,n_bytes";
	//! Buffered lines are joined in blocks of this many first, so a 600-row chunk is not built by 600 growing copies
	static const int JOIN_BLOCK = 32;

	protected string m_sName;
	protected string m_sDir;						//!< <session>/<stream>
	protected string m_sCommitPath;
	protected string m_sHeaderLine;					//!< header row + "\n"
	protected int m_iColumnCount;
	protected int m_iMaxRows = 1;
	protected int m_iMaxMs;
	protected bool m_bKeepCommitOpen;
	protected bool m_bVerifyLength;
	protected ref FileHandle m_CommitFile;			//!< only with keepCommitOpen
	protected int m_iCommitBytes;					//!< commit.csv bytes written so far (verifyLength)
	protected bool m_bOpen;
	protected bool m_bClosed;
	protected bool m_bFailed;
	protected string m_sFailure;

	// buffered chunk
	protected ref array<string> m_aLines = {};		//!< data rows, each ending with "\n"
	protected int m_iBufferFirstKey;
	protected int m_iBufferLastKey;
	protected int m_iBufferStartMs;

	// acknowledged (committed) data
	protected int m_iNextChunkId;
	protected int m_iCommittedRows;
	protected int m_iCommittedBytes;
	protected int m_iLastCommittedKey = -1;

	//------------------------------------------------------------------------------------------------
	//! Creates <sessionDir>/<name>/ and its commit.csv with the header (an existing commit.csv is refused).
	//! \param[in] maxRows chunk_max_rows of the stream (dc-2.0.streams.json)
	//! \param[in] maxMs clocks.chunk_max_s in ms
	//! \return false (GetFailure) on any failure
	bool Open(string sessionDir, string name, notnull array<string> columns, int maxRows, int maxMs, bool keepCommitOpen, bool verifyLength)
	{
		if (m_bOpen)
			return Fail("Open called twice for " + m_sName);

		m_bOpen = true;
		m_sName = name;
		m_sDir = sessionDir + "/" + name;
		m_sCommitPath = m_sDir + "/" + COMMIT_NAME;
		m_iColumnCount = columns.Count();
		m_iMaxRows = Math.Max(maxRows, 1);
		m_iMaxMs = maxMs;
		m_bKeepCommitOpen = keepCommitOpen;
		m_bVerifyLength = verifyLength;
		if (m_iColumnCount == 0)
			return Fail("no columns for " + name);

		string header;
		for (int i = 0; i < m_iColumnCount; i++)
		{
			if (i > 0)
				header += ",";

			header += columns[i];
		}

		m_sHeaderLine = header + "\n";

		// MakeDirectory's return for an existing directory is not documented: the directory must exist afterwards
		FileIO.MakeDirectory(m_sDir);
		if (!FileIO.FileExists(m_sDir))
			return Fail("cannot create the stream directory " + m_sDir);

		if (FileIO.FileExists(m_sCommitPath))
			return Fail("refusing to overwrite the existing " + m_sCommitPath);

		FileHandle file = FileIO.OpenFile(m_sCommitPath, FileMode.WRITE);
		if (!file)
			return Fail("FileIO.OpenFile(" + m_sCommitPath + ", WRITE) returned null");

		string data = COMMIT_HEADER + "\n";
		int expected = data.Length();
		int written = file.Write(data);
		if (written != expected)
		{
			file.Close();
			return Fail(string.Format("short or failed write of the %1 header: FileHandle.Write returned %2 of %3 bytes", m_sCommitPath, written, expected));
		}

		m_iCommitBytes = expected;
		if (m_bKeepCommitOpen)
			m_CommitFile = file;
		else
			file.Close();

		return VerifyCommitLength();
	}

	//------------------------------------------------------------------------------------------------
	override string GetName()
	{
		return m_sName;
	}

	//------------------------------------------------------------------------------------------------
	//! Buffers one row (its values joined with ','); commits the chunk when it holds chunk_max_rows rows.
	//! \return false on a failed write, a closed or failed writer or a row with the wrong number of values
	override bool AddRow(notnull DC_ProbeRow row, int key, int worldMs)
	{
		if (m_bFailed || m_bClosed || !m_bOpen)
			return false;

		if (row.Count() != m_iColumnCount)
			return Fail(string.Format("row with %1 values for %2 columns in %3", row.Count(), m_iColumnCount, m_sName));

		if (m_aLines.IsEmpty())
		{
			m_iBufferFirstKey = key;
			m_iBufferStartMs = worldMs;
		}

		m_iBufferLastKey = key;
		m_aLines.Insert(row.Join() + "\n");
		if (m_aLines.Count() >= m_iMaxRows)
			return Flush();

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Commits the buffered chunk once clocks.chunk_max_s of world time passed since its first row
	override bool FlushIfDue(int worldMs)
	{
		if (m_bFailed)
			return false;

		if (m_aLines.IsEmpty() || worldMs - m_iBufferStartMs < m_iMaxMs)
			return true;

		return Flush();
	}

	//------------------------------------------------------------------------------------------------
	//! Writes the buffered rows as the next chunk file (one Write), closes it, then appends its commit line.
	//! \return true when nothing was buffered or the chunk is acknowledged
	bool Flush()
	{
		if (m_bFailed)
			return false;

		int rows = m_aLines.Count();
		if (rows == 0)
			return true;

		string path = m_sDir + "/chunk_" + m_iNextChunkId.ToString(5) + ".csv";
		if (FileIO.FileExists(path))
			return Fail("refusing to overwrite the existing chunk " + path);

		string text = m_sHeaderLine + JoinLines(m_aLines);
		int expected = text.Length();
		FileHandle file = FileIO.OpenFile(path, FileMode.WRITE);
		if (!file)
			return Fail("FileIO.OpenFile(" + path + ", WRITE) returned null");

		int written = file.Write(text);
		file.Close();
		if (written != expected)
			return Fail(string.Format("short or failed write of %1: FileHandle.Write returned %2 of %3 bytes", path, written, expected));

		if (m_bVerifyLength)
		{
			int lengthClosed = DC_Session.ReadFileLength(path);
			if (lengthClosed != expected)
				return Fail(string.Format("%1 has %2 bytes after the close, %3 were written", path, lengthClosed, expected));
		}

		string commitLine = string.Format("%1,%2,%3,%4,%5", m_iNextChunkId, m_iBufferFirstKey, m_iBufferLastKey, rows, expected);
		if (!AppendCommitLine(commitLine))
			return false;

		m_iNextChunkId++;
		m_iCommittedRows += rows;
		m_iCommittedBytes += expected;
		m_iLastCommittedKey = m_iBufferLastKey;
		m_aLines.Clear();
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! One commit line: an open-APPEND-write-close cycle, or a Write through the handle kept open (TBD-3); with
	//! verifyLength the log's length is read back before the line counts
	protected bool AppendCommitLine(string line)
	{
		string data = line + "\n";
		int expected = data.Length();
		if (m_bKeepCommitOpen)
		{
			if (!m_CommitFile)
				return Fail("the commit log handle of " + m_sName + " is closed");

			int keptWritten = m_CommitFile.Write(data);
			if (keptWritten != expected)
				return Fail(string.Format("short or failed write of a %1 line: FileHandle.Write returned %2 of %3 bytes", m_sCommitPath, keptWritten, expected));

			m_iCommitBytes += expected;
			return VerifyCommitLength();
		}

		FileHandle file = FileIO.OpenFile(m_sCommitPath, FileMode.APPEND);
		if (!file)
			return Fail("FileIO.OpenFile(" + m_sCommitPath + ", APPEND) returned null");

		int written = file.Write(data);
		file.Close();
		if (written != expected)
			return Fail(string.Format("short or failed write of a %1 line: FileHandle.Write returned %2 of %3 bytes", m_sCommitPath, written, expected));

		m_iCommitBytes += expected;
		return VerifyCommitLength();
	}

	//------------------------------------------------------------------------------------------------
	//! TBD-3 fallback (verifyLength): commit.csv must hold exactly the bytes written so far, read back through the
	//! handle kept open (its GetLength; E-19 c03 records what it reports) or a READ handle on the closed file
	protected bool VerifyCommitLength()
	{
		if (!m_bVerifyLength)
			return true;

		int length;
		if (m_CommitFile)
			length = m_CommitFile.GetLength();
		else
			length = DC_Session.ReadFileLength(m_sCommitPath);

		if (length == m_iCommitBytes)
			return true;

		return Fail(string.Format("%1 has %2 bytes, %3 were written (TBD-3 length check)", m_sCommitPath, length, m_iCommitBytes));
	}

	//------------------------------------------------------------------------------------------------
	//! Commits what is buffered and closes the commit log (a log kept open is read back once more after its close
	//! with verifyLength). Idempotent. \return false when any write failed
	override bool Close()
	{
		if (m_bClosed)
			return !m_bFailed;

		if (m_bOpen && !m_bFailed)
			Flush();

		m_bClosed = true;
		if (!m_CommitFile)
			return !m_bFailed;

		CloseCommitFile();
		if (!m_bVerifyLength || m_bFailed)
			return !m_bFailed;

		int lengthClosed = DC_Session.ReadFileLength(m_sCommitPath);
		if (lengthClosed != m_iCommitBytes)
			Fail(string.Format("%1 has %2 bytes after its close, %3 were written (TBD-3 length check)", m_sCommitPath, lengthClosed, m_iCommitBytes));

		return !m_bFailed;
	}

	//------------------------------------------------------------------------------------------------
	//! Drops the buffered rows unwritten (never acknowledged); the writer stays open
	override void DiscardBuffered()
	{
		m_aLines.Clear();
	}

	//------------------------------------------------------------------------------------------------
	//! Drops the buffered rows unwritten (never acknowledged) and closes. Idempotent
	override void Abandon()
	{
		m_aLines.Clear();
		m_bClosed = true;
		CloseCommitFile();
	}

	//------------------------------------------------------------------------------------------------
	protected void CloseCommitFile()
	{
		if (!m_CommitFile)
			return;

		m_CommitFile.Close();
		m_CommitFile = null;
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
	override bool HasFailed()
	{
		return m_bFailed;
	}

	//------------------------------------------------------------------------------------------------
	override string GetFailure()
	{
		return m_sFailure;
	}

	//------------------------------------------------------------------------------------------------
	override int GetCommittedRows()
	{
		return m_iCommittedRows;
	}

	//------------------------------------------------------------------------------------------------
	override int GetCommittedChunks()
	{
		return m_iNextChunkId;
	}

	//------------------------------------------------------------------------------------------------
	override int GetLastCommittedKey()
	{
		return m_iLastCommittedKey;
	}

	//------------------------------------------------------------------------------------------------
	//! Rows buffered and not yet acknowledged
	int GetBufferedRows()
	{
		return m_aLines.Count();
	}

	//------------------------------------------------------------------------------------------------
	//! The lines concatenated in blocks of JOIN_BLOCK first, then the blocks
	protected static string JoinLines(notnull array<string> lines)
	{
		string text;
		string block;
		int inBlock = 0;
		foreach (string line : lines)
		{
			block += line;
			inBlock++;
			if (inBlock < JOIN_BLOCK)
				continue;

			text += block;
			block = string.Empty;
			inBlock = 0;
		}

		return text + block;
	}
}
