using System.Text;

namespace Axol.Core;

/// <summary>
/// Converts indentation-based AXOL (.axoli) to standard S-expression AXOL.
/// Lines not starting with '(' are auto-wrapped: keyword args... → (keyword args...)
/// Indented lines become children of the parent form (parent's closing ')' is deferred).
/// Lines starting with '(' are passed through as-is (assumed balanced).
/// </summary>
public static class IndentPreprocessor
{
    public static string Process(string source)
    {
        var lines = ParseLines(source);
        if (lines.Count == 0) return "";

        var sb = new StringBuilder();
        // Stack of indent levels that have unclosed forms
        var openStack = new Stack<int>();

        for (int i = 0; i < lines.Count; i++)
        {
            var (indent, content) = lines[i];

            // Close any forms whose indent level >= current (we've dedented past them)
            while (openStack.Count > 0 && indent <= openStack.Peek())
            {
                sb.Append(')');
                openStack.Pop();
            }

            bool hasChildren = i + 1 < lines.Count && lines[i + 1].indent > indent;

            if (content.StartsWith("("))
            {
                // Already an S-expression — append as-is
                if (sb.Length > 0) sb.Append(' ');
                sb.Append(content);
            }
            else
            {
                // Open a new form
                if (sb.Length > 0) sb.Append(' ');
                sb.Append('(').Append(content);

                if (hasChildren)
                {
                    // Leave open — children will follow
                    openStack.Push(indent);
                }
                else
                {
                    // Self-closing — no children
                    sb.Append(')');
                }
            }
        }

        // Close all remaining open forms
        while (openStack.Count > 0)
        {
            sb.Append(')');
            openStack.Pop();
        }

        return sb.ToString();
    }

    private static List<(int indent, string content)> ParseLines(string source)
    {
        var result = new List<(int indent, string content)>();
        var rawLines = source.Split('\n');

        foreach (var rawLine in rawLines)
        {
            var line = rawLine.TrimEnd('\r');
            if (string.IsNullOrWhiteSpace(line)) continue;

            int pos = 0;
            int indent = 0;
            while (pos < line.Length && (line[pos] == ' ' || line[pos] == '\t'))
            {
                indent += line[pos] == '\t' ? 2 : 1;
                pos++;
            }

            var content = line.Substring(pos).TrimEnd();

            // Skip comment-only lines
            if (content.StartsWith(";")) continue;

            // Handle inline comments
            var commentIdx = content.IndexOf(';');
            if (commentIdx > 0) content = content.Substring(0, commentIdx).TrimEnd();

            if (content.Length > 0)
                result.Add((indent, content));
        }

        return result;
    }
}
