using Axol.Interpreter.Values;

namespace Axol.Interpreter.BuiltinModules;

internal static class IoBuiltins
{
    public static void Register(Environment env)
    {
        env.Define("read_file", new BuiltinFunctionVal("read_file", args =>
        {
            if (args[0] is StrVal path)
            {
                try
                {
                    var content = File.ReadAllText(path.Value);
                    return new StrVal(content);
                }
                catch (Exception ex)
                {
                    throw new AxolRuntimeException($"read_file: {ex.Message}");
                }
            }
            throw new AxolRuntimeException("read_file: expected string path");
        }));

        env.Define("write_file", new BuiltinFunctionVal("write_file", args =>
        {
            if (args[0] is StrVal path && args[1] is StrVal content)
            {
                try
                {
                    File.WriteAllText(path.Value, content.Value);
                    return UnitVal.Instance;
                }
                catch (Exception ex)
                {
                    throw new AxolRuntimeException($"write_file: {ex.Message}");
                }
            }
            throw new AxolRuntimeException("write_file: expected (write_file path content)");
        }));
    }
}
