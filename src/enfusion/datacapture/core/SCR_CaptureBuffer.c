// ============================================================================
// SCR_CaptureBuffer - High-Performance Ring Buffer for Async I/O
// ============================================================================
//
// Lock-free ring buffer implementation for buffering captured data before
// async writes to disk. Designed for high-throughput data capture scenarios.
//
// FEATURES:
// - Fixed-size ring buffer with configurable capacity
// - Non-blocking writes with overflow handling
// - Batch flush support for efficient I/O
// - Statistics tracking for monitoring
// - Multiple serialization format support
//
// MEMORY LAYOUT:
// [ Slot 0 ][ Slot 1 ][ Slot 2 ] ... [ Slot N-1 ]
//     ^                    ^
//   head                  tail
//  (write)               (read)
//
// ============================================================================

// -----------------------------------------------------------------------------
// Buffer overflow policies
// -----------------------------------------------------------------------------
enum SCR_BufferOverflowPolicy
{
    OVERFLOW_DROP_OLDEST,       // Overwrite oldest entries (default)
    OVERFLOW_DROP_NEWEST,       // Reject new entries when full
    OVERFLOW_BLOCK,             // Block until space available (not recommended)
    OVERFLOW_EXPAND             // Dynamically expand buffer (memory risk)
}

// -----------------------------------------------------------------------------
// Buffer statistics
// -----------------------------------------------------------------------------
class SCR_BufferStats
{
    int m_iTotalWrites;
    int m_iTotalReads;
    int m_iDroppedWrites;
    int m_iFlushCount;
    int m_iPeakOccupancy;
    float m_fTotalWriteTimeMs;
    float m_fTotalFlushTimeMs;
    float m_fLastFlushTimeMs;

    //------------------------------------------------------------------------
    void SCR_BufferStats()
    {
        Reset();
    }

    void Reset()
    {
        m_iTotalWrites = 0;
        m_iTotalReads = 0;
        m_iDroppedWrites = 0;
        m_iFlushCount = 0;
        m_iPeakOccupancy = 0;
        m_fTotalWriteTimeMs = 0;
        m_fTotalFlushTimeMs = 0;
        m_fLastFlushTimeMs = 0;
    }

    //------------------------------------------------------------------------
    float GetAverageWriteTimeMs()
    {
        if (m_iTotalWrites == 0)
            return 0;
        return m_fTotalWriteTimeMs / m_iTotalWrites;
    }

    float GetAverageFlushTimeMs()
    {
        if (m_iFlushCount == 0)
            return 0;
        return m_fTotalFlushTimeMs / m_iFlushCount;
    }

    float GetDropRate()
    {
        int totalAttempts = m_iTotalWrites + m_iDroppedWrites;
        if (totalAttempts == 0)
            return 0;
        return (float)m_iDroppedWrites / totalAttempts;
    }
}

// -----------------------------------------------------------------------------
// SCR_BufferSlot - Individual slot in the ring buffer
// -----------------------------------------------------------------------------
class SCR_BufferSlot
{
    protected ref SCR_CaptureDataRecord m_Record;
    protected bool m_bOccupied;
    protected float m_fWriteTimeMs;

    //------------------------------------------------------------------------
    void SCR_BufferSlot()
    {
        m_Record = null;
        m_bOccupied = false;
        m_fWriteTimeMs = 0;
    }

    //------------------------------------------------------------------------
    void Write(SCR_CaptureDataRecord record, float timeMs)
    {
        m_Record = record;
        m_bOccupied = true;
        m_fWriteTimeMs = timeMs;
    }

    SCR_CaptureDataRecord Read()
    {
        SCR_CaptureDataRecord record = m_Record;
        m_Record = null;
        m_bOccupied = false;
        return record;
    }

    void Clear()
    {
        m_Record = null;
        m_bOccupied = false;
    }

    bool IsOccupied() { return m_bOccupied; }
    SCR_CaptureDataRecord Peek() { return m_Record; }
    float GetWriteTimeMs() { return m_fWriteTimeMs; }
}

// -----------------------------------------------------------------------------
// SCR_CaptureBuffer - Ring Buffer Implementation
// -----------------------------------------------------------------------------
class SCR_CaptureBuffer
{
    // Buffer configuration
    protected int m_iCapacity;
    protected SCR_BufferOverflowPolicy m_eOverflowPolicy;
    protected int m_iFlushThreshold;        // Flush when this many records buffered
    protected float m_fFlushIntervalMs;     // Max time between flushes

    // Ring buffer state
    protected ref array<ref SCR_BufferSlot> m_aSlots;
    protected int m_iHead;                  // Write position
    protected int m_iTail;                  // Read position
    protected int m_iCount;                 // Current occupancy

    // Timing
    protected float m_fLastFlushTimeMs;
    protected float m_fCreationTimeMs;

    // Statistics
    protected ref SCR_BufferStats m_Stats;

    // Serializer reference
    protected ref SCR_CaptureSerializer m_Serializer;

    // Constants
    static const int DEFAULT_CAPACITY = 1024;
    static const int MIN_CAPACITY = 16;
    static const int MAX_CAPACITY = 65536;
    static const int DEFAULT_FLUSH_THRESHOLD = 256;
    static const float DEFAULT_FLUSH_INTERVAL_MS = 5000;

    //------------------------------------------------------------------------
    void SCR_CaptureBuffer(
        int capacity = DEFAULT_CAPACITY,
        SCR_BufferOverflowPolicy overflowPolicy = SCR_BufferOverflowPolicy.OVERFLOW_DROP_OLDEST,
        int flushThreshold = DEFAULT_FLUSH_THRESHOLD,
        float flushIntervalMs = DEFAULT_FLUSH_INTERVAL_MS)
    {
        // Clamp capacity
        m_iCapacity = Math.Clamp(capacity, MIN_CAPACITY, MAX_CAPACITY);
        m_eOverflowPolicy = overflowPolicy;
        m_iFlushThreshold = Math.Clamp(flushThreshold, 1, m_iCapacity);
        m_fFlushIntervalMs = flushIntervalMs;

        // Initialize slots
        m_aSlots = new array<ref SCR_BufferSlot>();
        for (int i = 0; i < m_iCapacity; i++)
        {
            m_aSlots.Insert(new SCR_BufferSlot());
        }

        // Initialize state
        m_iHead = 0;
        m_iTail = 0;
        m_iCount = 0;
        m_fLastFlushTimeMs = 0;
        m_fCreationTimeMs = 0;

        // Initialize stats
        m_Stats = new SCR_BufferStats();

        Print("[CaptureBuffer] Initialized with capacity=" + m_iCapacity.ToString() +
              ", flushThreshold=" + m_iFlushThreshold.ToString(), LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Attach serializer for flush operations
    void SetSerializer(SCR_CaptureSerializer serializer)
    {
        m_Serializer = serializer;
    }

    //------------------------------------------------------------------------
    // Write a record to the buffer
    // Returns: true if write succeeded, false if dropped
    bool Write(SCR_CaptureDataRecord record, float currentTimeMs)
    {
        if (!record)
            return false;

        // Check if buffer is full
        if (m_iCount >= m_iCapacity)
        {
            switch (m_eOverflowPolicy)
            {
                case SCR_BufferOverflowPolicy.OVERFLOW_DROP_OLDEST:
                    // Advance tail to make room
                    AdvanceTail();
                    break;

                case SCR_BufferOverflowPolicy.OVERFLOW_DROP_NEWEST:
                    m_Stats.m_iDroppedWrites++;
                    return false;

                case SCR_BufferOverflowPolicy.OVERFLOW_BLOCK:
                    // Force flush and retry
                    Flush(currentTimeMs);
                    if (m_iCount >= m_iCapacity)
                    {
                        m_Stats.m_iDroppedWrites++;
                        return false;
                    }
                    break;

                case SCR_BufferOverflowPolicy.OVERFLOW_EXPAND:
                    if (!ExpandBuffer())
                    {
                        m_Stats.m_iDroppedWrites++;
                        return false;
                    }
                    break;
            }
        }

        // Write to head slot
        float writeStartMs = currentTimeMs;
        m_aSlots[m_iHead].Write(record, currentTimeMs);

        // Advance head
        m_iHead = (m_iHead + 1) % m_iCapacity;
        m_iCount++;

        // Update stats
        m_Stats.m_iTotalWrites++;
        if (m_iCount > m_Stats.m_iPeakOccupancy)
            m_Stats.m_iPeakOccupancy = m_iCount;

        // Check if we should auto-flush
        CheckAutoFlush(currentTimeMs);

        return true;
    }

    //------------------------------------------------------------------------
    // Read a single record from the buffer (FIFO)
    SCR_CaptureDataRecord Read()
    {
        if (m_iCount == 0)
            return null;

        SCR_CaptureDataRecord record = m_aSlots[m_iTail].Read();
        m_iTail = (m_iTail + 1) % m_iCapacity;
        m_iCount--;

        m_Stats.m_iTotalReads++;
        return record;
    }

    //------------------------------------------------------------------------
    // Peek at the next record without removing it
    SCR_CaptureDataRecord Peek()
    {
        if (m_iCount == 0)
            return null;
        return m_aSlots[m_iTail].Peek();
    }

    //------------------------------------------------------------------------
    // Read multiple records at once (batch read)
    int ReadBatch(array<ref SCR_CaptureDataRecord> outRecords, int maxCount)
    {
        if (!outRecords)
            return 0;

        int count = 0;
        int toRead = Math.Min(maxCount, m_iCount);

        for (int i = 0; i < toRead; i++)
        {
            SCR_CaptureDataRecord record = Read();
            if (record)
            {
                outRecords.Insert(record);
                count++;
            }
        }

        return count;
    }

    //------------------------------------------------------------------------
    // Flush all buffered records to serializer
    int Flush(float currentTimeMs)
    {
        if (!m_Serializer || m_iCount == 0)
            return 0;

        float flushStartMs = currentTimeMs;
        int flushedCount = 0;

        // Read all records and write to serializer
        while (m_iCount > 0)
        {
            SCR_CaptureDataRecord record = Read();
            if (record)
            {
                SCR_CaptureResult result = m_Serializer.WriteRecord(record);
                if (result.IsSuccess())
                    flushedCount++;
            }
        }

        // Update stats
        m_Stats.m_iFlushCount++;
        float flushDuration = currentTimeMs - flushStartMs;
        m_Stats.m_fTotalFlushTimeMs += flushDuration;
        m_Stats.m_fLastFlushTimeMs = flushDuration;
        m_fLastFlushTimeMs = currentTimeMs;

        return flushedCount;
    }

    //------------------------------------------------------------------------
    // Force flush if thresholds met
    protected void CheckAutoFlush(float currentTimeMs)
    {
        if (!m_Serializer)
            return;

        bool shouldFlush = false;

        // Check count threshold
        if (m_iCount >= m_iFlushThreshold)
            shouldFlush = true;

        // Check time threshold
        if (m_fLastFlushTimeMs > 0 && (currentTimeMs - m_fLastFlushTimeMs) >= m_fFlushIntervalMs)
            shouldFlush = true;

        if (shouldFlush)
            Flush(currentTimeMs);
    }

    //------------------------------------------------------------------------
    // Advance tail (discard oldest record)
    protected void AdvanceTail()
    {
        if (m_iCount == 0)
            return;

        m_aSlots[m_iTail].Clear();
        m_iTail = (m_iTail + 1) % m_iCapacity;
        m_iCount--;
        m_Stats.m_iDroppedWrites++;
    }

    //------------------------------------------------------------------------
    // Expand buffer (for OVERFLOW_EXPAND policy)
    protected bool ExpandBuffer()
    {
        int newCapacity = m_iCapacity * 2;
        if (newCapacity > MAX_CAPACITY)
            return false;

        // Create new slots
        ref array<ref SCR_BufferSlot> newSlots = new array<ref SCR_BufferSlot>();

        // Copy existing data in order
        int readPos = m_iTail;
        for (int i = 0; i < m_iCount; i++)
        {
            newSlots.Insert(m_aSlots[readPos]);
            readPos = (readPos + 1) % m_iCapacity;
        }

        // Add new empty slots
        for (int i = m_iCount; i < newCapacity; i++)
        {
            newSlots.Insert(new SCR_BufferSlot());
        }

        // Update state
        m_aSlots = newSlots;
        m_iTail = 0;
        m_iHead = m_iCount;
        m_iCapacity = newCapacity;

        Print("[CaptureBuffer] Expanded to capacity=" + m_iCapacity.ToString(), LogLevel.WARNING);
        return true;
    }

    //------------------------------------------------------------------------
    // Clear all buffered data
    void Clear()
    {
        for (int i = 0; i < m_iCapacity; i++)
        {
            m_aSlots[i].Clear();
        }
        m_iHead = 0;
        m_iTail = 0;
        m_iCount = 0;
    }

    //------------------------------------------------------------------------
    // Accessors
    int GetCapacity() { return m_iCapacity; }
    int GetCount() { return m_iCount; }
    int GetFreeSpace() { return m_iCapacity - m_iCount; }
    bool IsEmpty() { return m_iCount == 0; }
    bool IsFull() { return m_iCount >= m_iCapacity; }
    float GetOccupancyPercent() { return (float)m_iCount / m_iCapacity * 100.0; }

    SCR_BufferOverflowPolicy GetOverflowPolicy() { return m_eOverflowPolicy; }
    void SetOverflowPolicy(SCR_BufferOverflowPolicy policy) { m_eOverflowPolicy = policy; }

    int GetFlushThreshold() { return m_iFlushThreshold; }
    void SetFlushThreshold(int threshold)
    {
        m_iFlushThreshold = Math.Clamp(threshold, 1, m_iCapacity);
    }

    float GetFlushIntervalMs() { return m_fFlushIntervalMs; }
    void SetFlushIntervalMs(float intervalMs) { m_fFlushIntervalMs = intervalMs; }

    //------------------------------------------------------------------------
    // Statistics
    SCR_BufferStats GetStats() { return m_Stats; }

    void ResetStats()
    {
        m_Stats.Reset();
    }

    //------------------------------------------------------------------------
    // Debug output
    string GetDebugString()
    {
        string str = "[Buffer] cap=" + m_iCapacity.ToString();
        str += " count=" + m_iCount.ToString();
        str += " (" + GetOccupancyPercent().ToString(5, 1) + "%)";
        str += " head=" + m_iHead.ToString();
        str += " tail=" + m_iTail.ToString();
        str += " dropped=" + m_Stats.m_iDroppedWrites.ToString();
        return str;
    }

    //------------------------------------------------------------------------
    // Finalize and flush remaining
    int Finalize(float currentTimeMs)
    {
        int flushed = Flush(currentTimeMs);
        Clear();
        return flushed;
    }
}

// -----------------------------------------------------------------------------
// SCR_MultiBuffer - Multiple buffers for different data types
// -----------------------------------------------------------------------------
class SCR_MultiBuffer
{
    protected ref map<string, ref SCR_CaptureBuffer> m_mBuffers;
    protected int m_iDefaultCapacity;
    protected SCR_BufferOverflowPolicy m_eDefaultPolicy;

    //------------------------------------------------------------------------
    void SCR_MultiBuffer(
        int defaultCapacity = SCR_CaptureBuffer.DEFAULT_CAPACITY,
        SCR_BufferOverflowPolicy defaultPolicy = SCR_BufferOverflowPolicy.OVERFLOW_DROP_OLDEST)
    {
        m_mBuffers = new map<string, ref SCR_CaptureBuffer>();
        m_iDefaultCapacity = defaultCapacity;
        m_eDefaultPolicy = defaultPolicy;
    }

    //------------------------------------------------------------------------
    // Get or create buffer for module
    SCR_CaptureBuffer GetBuffer(string moduleId)
    {
        if (!m_mBuffers.Contains(moduleId))
        {
            SCR_CaptureBuffer buffer = new SCR_CaptureBuffer(m_iDefaultCapacity, m_eDefaultPolicy);
            m_mBuffers.Set(moduleId, buffer);
        }
        return m_mBuffers.Get(moduleId);
    }

    //------------------------------------------------------------------------
    // Set specific buffer for module
    void SetBuffer(string moduleId, SCR_CaptureBuffer buffer)
    {
        m_mBuffers.Set(moduleId, buffer);
    }

    //------------------------------------------------------------------------
    // Flush all buffers
    int FlushAll(float currentTimeMs)
    {
        int totalFlushed = 0;
        foreach (string moduleId, SCR_CaptureBuffer buffer : m_mBuffers)
        {
            if (buffer)
                totalFlushed += buffer.Flush(currentTimeMs);
        }
        return totalFlushed;
    }

    //------------------------------------------------------------------------
    // Clear all buffers
    void ClearAll()
    {
        foreach (string moduleId, SCR_CaptureBuffer buffer : m_mBuffers)
        {
            if (buffer)
                buffer.Clear();
        }
    }

    //------------------------------------------------------------------------
    // Get total count across all buffers
    int GetTotalCount()
    {
        int total = 0;
        foreach (string moduleId, SCR_CaptureBuffer buffer : m_mBuffers)
        {
            if (buffer)
                total += buffer.GetCount();
        }
        return total;
    }

    //------------------------------------------------------------------------
    // Get buffer count
    int GetBufferCount()
    {
        return m_mBuffers.Count();
    }

    //------------------------------------------------------------------------
    // Finalize all
    int FinalizeAll(float currentTimeMs)
    {
        int totalFlushed = 0;
        foreach (string moduleId, SCR_CaptureBuffer buffer : m_mBuffers)
        {
            if (buffer)
                totalFlushed += buffer.Finalize(currentTimeMs);
        }
        return totalFlushed;
    }
}

// Forward declaration
class SCR_CaptureSerializer;
