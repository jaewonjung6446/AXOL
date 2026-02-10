namespace Axol.Core;

public sealed class SourceMap
{
    private readonly string _source;
    private readonly int[] _lineStarts;

    public SourceMap(string source)
    {
        _source = source;
        _lineStarts = ComputeLineStarts(source);
    }

    public (int Line, int Col) GetLineCol(int offset)
    {
        if (offset < 0) offset = 0;
        if (offset > _source.Length) offset = _source.Length;

        int lo = 0, hi = _lineStarts.Length - 1;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) / 2;
            if (_lineStarts[mid] <= offset)
                lo = mid;
            else
                hi = mid - 1;
        }

        return (lo + 1, offset - _lineStarts[lo] + 1);
    }

    private static int[] ComputeLineStarts(string source)
    {
        var starts = new List<int> { 0 };
        for (int i = 0; i < source.Length; i++)
        {
            if (source[i] == '\n')
                starts.Add(i + 1);
            else if (source[i] == '\r')
            {
                if (i + 1 < source.Length && source[i + 1] == '\n')
                    i++;
                starts.Add(i + 1);
            }
        }
        return starts.ToArray();
    }
}
