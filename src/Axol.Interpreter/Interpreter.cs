using Axol.Core.Ast;
using Axol.Core.Diagnostics;
using Axol.Interpreter.Values;

namespace Axol.Interpreter;

public sealed class AxolInterpreter
{
    private readonly Environment _global;
    private readonly DiagnosticBag _diagnostics;
    private readonly TextWriter _output;
    private readonly ModuleRegistry _modules = new();

    public Environment GlobalEnv => _global;
    public DiagnosticBag Diagnostics => _diagnostics;
    public ModuleRegistry Modules => _modules;

    public AxolInterpreter(TextWriter? output = null, DiagnosticBag? diagnostics = null)
    {
        _output = output ?? Console.Out;
        _diagnostics = diagnostics ?? new DiagnosticBag();
        _global = new Environment();
        Builtins.Register(_global, _output, CallFunction);
    }

    public AxolValue Run(Program program)
    {
        AxolValue result = UnitVal.Instance;
        foreach (var form in program.Forms)
            result = Eval(form, _global);
        return result;
    }

    public AxolValue Eval(AstNode node, Environment env)
    {
        return node switch
        {
            NumberLitInt n => new IntVal(n.Value),
            NumberLitFloat n => new FloatVal(n.Value),
            StringLit s => new StrVal(s.Value),
            BoolLit b => new BoolVal(b.Value),
            NilLit => NilVal.Instance,
            SymbolRef sym => env.Get(sym.Name),
            ListForm form => EvalListForm(form, env),
            TypeAnnotation => UnitVal.Instance,
            Program p => EvalProgram(p, env),
            _ => throw new AxolRuntimeException($"Unknown node type: {node.GetType().Name}")
        };
    }

    private AxolValue EvalProgram(Program p, Environment env)
    {
        AxolValue result = UnitVal.Instance;
        foreach (var form in p.Forms)
            result = Eval(form, env);
        return result;
    }

    private AxolValue EvalListForm(ListForm form, Environment env)
    {
        return form.Keyword switch
        {
            "f" => EvalFuncDef(form, env),
            "v" => EvalLetImmutable(form, env),
            "m" => EvalLetMutable(form, env),
            "m!" => EvalMutate(form, env),
            "?" => EvalIf(form, env),
            "W" => EvalWhile(form, env),
            "F" => EvalFor(form, env),
            "X" => EvalMatch(form, env),
            "R" => EvalReturn(form, env),
            "D" => EvalDo(form, env),
            "L" => EvalLambda(form, env),
            "P" => EvalPipe(form, env),
            "@" => EvalFieldAccess(form, env),
            "#" => EvalIndexAccess(form, env),
            "S" => EvalStructLiteral(form, env),
            "A" => EvalArrayLiteral(form, env),
            "H" => EvalHashMapLiteral(form, env),
            "Q" => EvalPrecondition(form, env),
            "G" => EvalPostcondition(form, env),
            "!" => EvalAssert(form, env),
            "E" => EvalThrow(form, env),
            "C" => EvalCatch(form, env),
            "t" => EvalTypeDef(form, env),
            "e" => EvalEnumDef(form, env),
            "M" => EvalModule(form, env),
            "import" => EvalImport(form, env),
            "use" => EvalUse(form, env),
            "+" or "-" or "*" or "/" or "%" => EvalArithmetic(form, env),
            "=" or "!=" or "<" or ">" or "<=" or ">=" => EvalComparison(form, env),
            "&" or "|" or "~" => EvalLogical(form, env),
            _ => EvalFunctionCall(form, env)
        };
    }

    private AxolValue EvalFuncDef(ListForm form, Environment env)
    {
        int idx = 0;
        var args = form.Args;

        if (idx >= args.Count || args[idx] is not SymbolRef nameRef)
            throw new AxolRuntimeException("f: expected function name");
        var name = nameRef.Name;
        idx++;

        // skip optional type annotation
        if (idx < args.Count && args[idx] is TypeAnnotation)
            idx++;

        // collect parameter names
        var paramNames = new List<string>();
        while (idx < args.Count && args[idx] is SymbolRef pRef && !IsBodyStart(args, idx))
        {
            paramNames.Add(pRef.Name);
            idx++;
        }

        // remaining args = body forms
        var body = args.Skip(idx).ToList();

        var fn = new FunctionVal(name, paramNames, body, env);
        env.Define(name, fn);
        return fn;
    }

    private static bool IsBodyStart(IReadOnlyList<AstNode> args, int idx)
    {
        if (args[idx] is ListForm) return true;
        if (args[idx] is NumberLitInt or NumberLitFloat or StringLit or BoolLit or NilLit) return true;
        if (idx == args.Count - 1 && args[idx] is SymbolRef) return true;
        return false;
    }

    private AxolValue EvalLetImmutable(ListForm form, Environment env)
    {
        if (form.Args.Count < 2 || form.Args[0] is not SymbolRef nameRef)
            throw new AxolRuntimeException("v: expected (v name value)");
        var value = Eval(form.Args[1], env);
        env.Define(nameRef.Name, value, mutable: false);
        return value;
    }

    private AxolValue EvalLetMutable(ListForm form, Environment env)
    {
        if (form.Args.Count < 2 || form.Args[0] is not SymbolRef nameRef)
            throw new AxolRuntimeException("m: expected (m name value)");
        var value = Eval(form.Args[1], env);
        env.Define(nameRef.Name, value, mutable: true);
        return value;
    }

    private AxolValue EvalMutate(ListForm form, Environment env)
    {
        if (form.Args.Count < 2)
            throw new AxolRuntimeException("m!: expected (m! target value)");

        var value = Eval(form.Args[1], env);

        if (form.Args[0] is SymbolRef nameRef)
        {
            env.Set(nameRef.Name, value);
            return value;
        }

        if (form.Args[0] is ListForm access && access.Keyword == "@")
        {
            var obj = Eval(access.Args[0], env);
            if (access.Args[1] is not SymbolRef fieldRef)
                throw new AxolRuntimeException("m!: field name must be symbol");
            var fieldName = fieldRef.Name;

            if (obj is StructVal sv) { sv.Fields[fieldName] = value; return value; }
            if (obj is MapVal mv) { mv.Entries[fieldName] = value; return value; }
        }

        if (form.Args[0] is ListForm idxAccess && idxAccess.Keyword == "#")
        {
            var obj = Eval(idxAccess.Args[0], env);
            var idx = Eval(idxAccess.Args[1], env);
            if (obj is ListVal lv && idx is IntVal iv)
            {
                lv.Items[(int)iv.Value] = value;
                return value;
            }
        }

        throw new AxolRuntimeException("m!: invalid mutation target");
    }

    private AxolValue EvalIf(ListForm form, Environment env)
    {
        if (form.Args.Count < 2)
            throw new AxolRuntimeException("?: expected (? cond then) or (? cond then else)");

        var cond = Eval(form.Args[0], env);
        if (cond.IsTruthy)
            return Eval(form.Args[1], env);
        if (form.Args.Count >= 3)
            return Eval(form.Args[2], env);
        return NilVal.Instance;
    }

    private AxolValue EvalWhile(ListForm form, Environment env)
    {
        if (form.Args.Count < 2)
            throw new AxolRuntimeException("W: expected (W cond body...)");

        AxolValue result = UnitVal.Instance;
        while (true)
        {
            var cond = Eval(form.Args[0], env);
            if (!cond.IsTruthy) break;

            try
            {
                for (int i = 1; i < form.Args.Count; i++)
                    result = Eval(form.Args[i], env);
            }
            catch (BreakSignal) { break; }
            catch (ContinueSignal) { continue; }
        }
        return result;
    }

    private AxolValue EvalFor(ListForm form, Environment env)
    {
        if (form.Args.Count < 3 || form.Args[0] is not SymbolRef varRef)
            throw new AxolRuntimeException("F: expected (F var iterable body...)");

        var iterable = Eval(form.Args[1], env);
        if (iterable is not ListVal list)
            throw new AxolRuntimeException("F: iterable must be a list");

        AxolValue result = UnitVal.Instance;
        var loopEnv = env.CreateChild();

        foreach (var item in list.Items)
        {
            loopEnv.Define(varRef.Name, item, mutable: true);
            try
            {
                for (int i = 2; i < form.Args.Count; i++)
                    result = Eval(form.Args[i], loopEnv);
            }
            catch (BreakSignal) { break; }
            catch (ContinueSignal) { continue; }
        }
        return result;
    }

    // ─── Pattern Matching (Step 4) ───

    private AxolValue EvalMatch(ListForm form, Environment env)
    {
        if (form.Args.Count < 3)
            throw new AxolRuntimeException("X: expected (X subject pattern1 body1 ...)");

        var subject = Eval(form.Args[0], env);

        int i = 1;
        while (i + 1 < form.Args.Count)
        {
            var pattern = form.Args[i];
            i++;

            // Check for guard clause: (when condition)
            AstNode? guard = null;
            if (i < form.Args.Count && form.Args[i] is ListForm guardForm && guardForm.Keyword == "when")
            {
                guard = guardForm.Args.Count > 0 ? guardForm.Args[0] : null;
                i++;
            }

            if (i > form.Args.Count) break;
            var body = form.Args[i];
            i++;

            if (MatchPattern(subject, pattern, env, out var bindings))
            {
                var matchEnv = env.CreateChild();
                foreach (var kv in bindings)
                    matchEnv.Define(kv.Key, kv.Value);

                // Evaluate guard if present
                if (guard != null)
                {
                    var guardResult = Eval(guard, matchEnv);
                    if (!guardResult.IsTruthy)
                        continue;
                }

                return Eval(body, matchEnv);
            }
        }

        return NilVal.Instance;
    }

    private bool MatchPattern(AxolValue subject, AstNode pattern, Environment env, out Dictionary<string, AxolValue> bindings)
    {
        bindings = new Dictionary<string, AxolValue>();

        // Wildcard
        if (pattern is SymbolRef sym)
        {
            if (sym.Name == "_")
                return true;

            // Check if this is an enum variant reference (EnumName.Variant)
            if (sym.Name.Contains('.'))
            {
                if (env.TryGet(sym.Name, out var enumVal) && enumVal is EnumVariantVal ev)
                {
                    return subject is EnumVariantVal sv && sv.EnumName == ev.EnumName && sv.VariantName == ev.VariantName;
                }
            }

            bindings[sym.Name] = subject;
            return true;
        }

        // Literals
        if (pattern is NumberLitInt ni)
            return subject is IntVal iv && iv.Value == ni.Value;
        if (pattern is NumberLitFloat nf)
            return subject is FloatVal fv && Math.Abs(fv.Value - nf.Value) < double.Epsilon;
        if (pattern is StringLit sl)
            return subject is StrVal sv2 && sv2.Value == sl.Value;
        if (pattern is BoolLit bl)
            return subject is BoolVal bv && bv.Value == bl.Value;
        if (pattern is NilLit)
            return subject is NilVal;

        // List form patterns
        if (pattern is ListForm lf)
        {
            // Array destructuring: (A head rest...)
            if (lf.Keyword == "A")
                return MatchListDestructure(subject, lf, env, bindings);

            // Struct destructuring: (S TypeName field1 field2 ...)
            if (lf.Keyword == "S")
                return MatchStructDestructure(subject, lf, bindings);

            // Enum variant with data: (EnumName.Variant bindVar)
            // Recognized by dotted keyword: e.g. Shape.Circle
            if (lf.Keyword.Contains('.'))
                return MatchEnumVariant(subject, lf, env, bindings);

            // Desugared enum variant: (@ EnumName Variant) from dot access desugaring
            if (lf.Keyword == "@" && lf.Args.Count == 2)
            {
                var reconstructed = TryReconstructDottedName(lf);
                if (reconstructed != null && env.TryGet(reconstructed, out var enumVal) && enumVal is EnumVariantVal ev)
                    return subject is EnumVariantVal sv && sv.EnumName == ev.EnumName && sv.VariantName == ev.VariantName;
            }
        }

        return false;
    }

    private bool MatchListDestructure(AxolValue subject, ListForm pattern, Environment env, Dictionary<string, AxolValue> bindings)
    {
        if (subject is not ListVal list) return false;

        var patternArgs = pattern.Args;
        for (int i = 0; i < patternArgs.Count; i++)
        {
            // Rest pattern: name...
            if (patternArgs[i] is SymbolRef restSym && restSym.Name.EndsWith("..."))
            {
                var restName = restSym.Name[..^3]; // remove "..."
                var rest = list.Items.Skip(i).ToList();
                bindings[restName] = new ListVal(rest);
                return true;
            }

            if (i >= list.Items.Count) return false;

            if (!MatchPattern(list.Items[i], patternArgs[i], env, out var subBindings))
                return false;
            foreach (var kv in subBindings)
                bindings[kv.Key] = kv.Value;
        }

        // If pattern has fewer elements than list, still matches (partial)
        return patternArgs.Count <= list.Items.Count;
    }

    private static bool MatchStructDestructure(AxolValue subject, ListForm pattern, Dictionary<string, AxolValue> bindings)
    {
        if (subject is not StructVal sv) return false;
        if (pattern.Args.Count < 1 || pattern.Args[0] is not SymbolRef typeRef) return false;
        if (sv.TypeName != typeRef.Name) return false;

        // Remaining args are field names to bind
        for (int i = 1; i < pattern.Args.Count; i++)
        {
            if (pattern.Args[i] is SymbolRef fieldRef)
            {
                if (sv.Fields.TryGetValue(fieldRef.Name, out var fieldVal))
                    bindings[fieldRef.Name] = fieldVal;
                else
                    return false;
            }
        }
        return true;
    }

    private bool MatchEnumVariant(AxolValue subject, ListForm pattern, Environment env, Dictionary<string, AxolValue> bindings)
    {
        if (subject is not EnumVariantVal ev) return false;

        var qualName = pattern.Keyword; // e.g. "Shape.Circle"
        var parts = qualName.Split('.');
        if (parts.Length != 2) return false;

        if (ev.EnumName != parts[0] || ev.VariantName != parts[1])
            return false;

        // Bind data fields
        if (pattern.Args.Count > 0 && ev.Data is ListVal dataList)
        {
            for (int i = 0; i < pattern.Args.Count && i < dataList.Items.Count; i++)
            {
                if (pattern.Args[i] is SymbolRef argRef)
                    bindings[argRef.Name] = dataList.Items[i];
            }
        }
        else if (pattern.Args.Count == 1 && ev.Data != null)
        {
            // Single data value
            if (pattern.Args[0] is SymbolRef argRef)
                bindings[argRef.Name] = ev.Data;
        }

        return true;
    }

    // ─── Basic forms (continued) ───

    private AxolValue EvalReturn(ListForm form, Environment env)
    {
        var value = form.Args.Count > 0 ? Eval(form.Args[0], env) : UnitVal.Instance;
        throw new ReturnSignal(value);
    }

    private AxolValue EvalDo(ListForm form, Environment env)
    {
        var doEnv = env.CreateChild();
        AxolValue result = UnitVal.Instance;
        foreach (var arg in form.Args)
            result = Eval(arg, doEnv);
        return result;
    }

    private AxolValue EvalLambda(ListForm form, Environment env)
    {
        var paramNames = new List<string>();
        int bodyStart;

        if (form.Args[0] is ListForm paramList)
        {
            // The parser produces ListForm(keyword, args) for (a b c)
            // keyword = "a", args = [b, c] — so keyword is the first param
            paramNames.Add(paramList.Keyword);
            foreach (var p in paramList.Args)
            {
                if (p is SymbolRef sr)
                    paramNames.Add(sr.Name);
            }
            bodyStart = 1;
        }
        else if (form.Args[0] is SymbolRef singleParam)
        {
            paramNames.Add(singleParam.Name);
            bodyStart = 1;
        }
        else
        {
            bodyStart = 0;
        }

        var body = form.Args.Skip(bodyStart).ToList();
        return new FunctionVal("<lambda>", paramNames, body, env);
    }

    private AxolValue EvalPipe(ListForm form, Environment env)
    {
        if (form.Args.Count < 2)
            throw new AxolRuntimeException("P: expected (P value fn1 fn2 ...)");

        var value = Eval(form.Args[0], env);
        for (int i = 1; i < form.Args.Count; i++)
        {
            var fn = Eval(form.Args[i], env);
            value = CallFunction(fn, new List<AxolValue> { value });
        }
        return value;
    }

    private AxolValue EvalFieldAccess(ListForm form, Environment env)
    {
        if (form.Args.Count < 2)
            throw new AxolRuntimeException("@: expected (@ obj field)");

        // Try flat key lookup first (for module/enum names like math.square, Color.Red)
        var dotted = TryReconstructDottedName(form);
        if (dotted != null && env.TryGet(dotted, out var flatVal))
            return flatVal;

        var obj = Eval(form.Args[0], env);
        if (form.Args[1] is not SymbolRef fieldRef)
            throw new AxolRuntimeException("@: field name must be symbol");

        var fieldName = fieldRef.Name;

        return obj switch
        {
            StructVal sv => sv.Fields.TryGetValue(fieldName, out var fv) ? fv
                : throw new AxolRuntimeException($"@: struct {sv.TypeName} has no field '{fieldName}'"),
            MapVal mv => mv.Entries.TryGetValue(fieldName, out var mv2) ? mv2
                : throw new AxolRuntimeException($"@: map has no key '{fieldName}'"),
            _ => throw new AxolRuntimeException($"@: cannot access field on {obj.GetType().Name}")
        };
    }

    private static string? TryReconstructDottedName(AstNode node) => node switch
    {
        SymbolRef sr => sr.Name,
        ListForm lf when lf.Keyword == "@" && lf.Args.Count == 2 && lf.Args[1] is SymbolRef fieldSr =>
            TryReconstructDottedName(lf.Args[0]) is string prefix ? $"{prefix}.{fieldSr.Name}" : null,
        _ => null
    };

    private AxolValue EvalIndexAccess(ListForm form, Environment env)
    {
        if (form.Args.Count < 2)
            throw new AxolRuntimeException("#: expected (# collection index)");

        var collection = Eval(form.Args[0], env);
        var index = Eval(form.Args[1], env);

        return (collection, index) switch
        {
            (ListVal lv, IntVal iv) => iv.Value >= 0 && iv.Value < lv.Items.Count
                ? lv.Items[(int)iv.Value]
                : throw new AxolRuntimeException($"#: index {iv.Value} out of bounds"),
            (StrVal sv, IntVal iv) => iv.Value >= 0 && iv.Value < sv.Value.Length
                ? new StrVal(sv.Value[(int)iv.Value].ToString())
                : throw new AxolRuntimeException($"#: index {iv.Value} out of bounds"),
            (MapVal mv, StrVal sv) => mv.Entries.TryGetValue(sv.Value, out var val) ? val
                : throw new AxolRuntimeException($"#: map has no key '{sv.Value}'"),
            _ => throw new AxolRuntimeException("#: invalid collection/index types")
        };
    }

    private AxolValue EvalStructLiteral(ListForm form, Environment env)
    {
        if (form.Args.Count < 1 || form.Args[0] is not SymbolRef typeRef)
            throw new AxolRuntimeException("S: expected (S TypeName field1 val1 ...)");

        var fields = new Dictionary<string, AxolValue>();
        for (int i = 1; i + 1 < form.Args.Count; i += 2)
        {
            if (form.Args[i] is not SymbolRef fieldRef)
                throw new AxolRuntimeException("S: field name must be symbol");
            fields[fieldRef.Name] = Eval(form.Args[i + 1], env);
        }
        return new StructVal(typeRef.Name, fields);
    }

    private AxolValue EvalArrayLiteral(ListForm form, Environment env)
    {
        var items = form.Args.Select(a => Eval(a, env)).ToList();
        return new ListVal(items);
    }

    private AxolValue EvalHashMapLiteral(ListForm form, Environment env)
    {
        var entries = new Dictionary<string, AxolValue>();
        for (int i = 0; i + 1 < form.Args.Count; i += 2)
        {
            var key = Eval(form.Args[i], env);
            entries[key.Display()] = Eval(form.Args[i + 1], env);
        }
        return new MapVal(entries);
    }

    private AxolValue EvalPrecondition(ListForm form, Environment env)
    {
        if (form.Args.Count < 1)
            throw new AxolRuntimeException("Q: expected (Q condition)");
        var cond = Eval(form.Args[0], env);
        if (!cond.IsTruthy)
        {
            var exprStr = FormatNode(form.Args[0]);
            var json = $"{{\"loc\":[{form.Span.Start},{form.Span.End}],\"code\":\"E301\",\"msg\":\"precondition failed\",\"expr\":\"{EscapeJson(exprStr)}\"}}";
            throw new ContractViolationException($"Precondition failed: {exprStr}", json);
        }
        return UnitVal.Instance;
    }

    private AxolValue EvalPostcondition(ListForm form, Environment env)
    {
        if (form.Args.Count < 1)
            throw new AxolRuntimeException("G: expected (G condition)");
        var cond = Eval(form.Args[0], env);
        if (!cond.IsTruthy)
        {
            var exprStr = FormatNode(form.Args[0]);
            var json = $"{{\"loc\":[{form.Span.Start},{form.Span.End}],\"code\":\"E302\",\"msg\":\"postcondition failed\",\"expr\":\"{EscapeJson(exprStr)}\"}}";
            throw new ContractViolationException($"Postcondition failed: {exprStr}", json);
        }
        return UnitVal.Instance;
    }

    private AxolValue EvalAssert(ListForm form, Environment env)
    {
        if (form.Args.Count < 1)
            throw new AxolRuntimeException("!: expected (! condition)");
        var cond = Eval(form.Args[0], env);
        if (!cond.IsTruthy)
        {
            var exprStr = FormatNode(form.Args[0]);
            var json = $"{{\"loc\":[{form.Span.Start},{form.Span.End}],\"code\":\"E303\",\"msg\":\"assertion failed\",\"expr\":\"{EscapeJson(exprStr)}\"}}";
            throw new AssertionFailedException($"Assertion failed: {exprStr}", json);
        }
        return UnitVal.Instance;
    }

    private AxolValue EvalThrow(ListForm form, Environment env)
    {
        var msg = form.Args.Count > 0 ? Eval(form.Args[0], env).Display() : "error";
        var json = $"{{\"loc\":[{form.Span.Start},{form.Span.End}],\"code\":\"E400\",\"msg\":\"{EscapeJson(msg)}\"}}";
        throw new AxolRuntimeException(msg, json);
    }

    private AxolValue EvalCatch(ListForm form, Environment env)
    {
        if (form.Args.Count < 3)
            throw new AxolRuntimeException("C: expected (C try_expr err_var handler_body)");

        try
        {
            return Eval(form.Args[0], env);
        }
        catch (AxolRuntimeException ex)
        {
            if (form.Args[1] is not SymbolRef errVar)
                throw new AxolRuntimeException("C: error variable must be symbol");
            var catchEnv = env.CreateChild();
            catchEnv.Define(errVar.Name, new StrVal(ex.Message));
            return Eval(form.Args[2], catchEnv);
        }
    }

    private AxolValue EvalTypeDef(ListForm form, Environment env)
    {
        if (form.Args.Count < 1 || form.Args[0] is not SymbolRef typeRef)
            throw new AxolRuntimeException("t: expected (t TypeName ...)");

        var typeName = typeRef.Name;
        var fieldNames = new List<string>();
        for (int i = 1; i < form.Args.Count; i++)
        {
            if (form.Args[i] is SymbolRef fieldSym)
                fieldNames.Add(fieldSym.Name);
        }

        var fields = fieldNames.ToList();
        env.Define(typeName, new BuiltinFunctionVal(typeName, args =>
        {
            var dict = new Dictionary<string, AxolValue>();
            for (int j = 0; j < fields.Count && j < args.Count; j++)
                dict[fields[j]] = args[j];
            return new StructVal(typeName, dict);
        }));
        return UnitVal.Instance;
    }

    // ─── Enum definitions (Step 4: data-bearing variants) ───

    private AxolValue EvalEnumDef(ListForm form, Environment env)
    {
        if (form.Args.Count < 1 || form.Args[0] is not SymbolRef enumRef)
            throw new AxolRuntimeException("e: expected (e EnumName variant1 variant2 ...)");

        var enumName = enumRef.Name;
        for (int i = 1; i < form.Args.Count; i++)
        {
            if (form.Args[i] is SymbolRef variantRef)
            {
                // Simple variant (no data)
                var vName = variantRef.Name;
                env.Define($"{enumName}.{vName}", new EnumVariantVal(enumName, vName, null));
            }
            else if (form.Args[i] is ListForm variantForm)
            {
                // Data-bearing variant: (VariantName field1 field2 ...)
                if (variantForm.Keyword is string vName)
                {
                    var paramCount = variantForm.Args.Count;
                    var capturedName = vName;
                    var capturedEnum = enumName;
                    if (paramCount == 1)
                    {
                        // Single data: constructor returns EnumVariantVal with single data
                        env.Define($"{enumName}.{vName}", new BuiltinFunctionVal($"{enumName}.{vName}", args =>
                            new EnumVariantVal(capturedEnum, capturedName, args[0])));
                    }
                    else
                    {
                        // Multiple data: constructor wraps in ListVal
                        env.Define($"{enumName}.{vName}", new BuiltinFunctionVal($"{enumName}.{vName}", args =>
                            new EnumVariantVal(capturedEnum, capturedName, new ListVal(args.ToList()))));
                    }
                }
            }
        }
        return UnitVal.Instance;
    }

    // ─── Module System (Step 5) ───

    private AxolValue EvalModule(ListForm form, Environment env)
    {
        if (form.Args.Count < 1 || form.Args[0] is not SymbolRef modNameRef)
            throw new AxolRuntimeException("M: expected (M ModName body...)");

        var modName = modNameRef.Name;
        var modEnv = env.CreateChild();

        AxolValue result = UnitVal.Instance;
        for (int i = 1; i < form.Args.Count; i++)
            result = Eval(form.Args[i], modEnv);

        // Register module
        _modules.Register(modName, modEnv);

        // Also define module bindings as ModName.binding in parent scope
        foreach (var kv in modEnv.GetAllBindings())
        {
            // Only export user-defined bindings (skip builtins by checking if already in env)
            if (!env.TryGet(kv.Key, out _))
                env.Define($"{modName}.{kv.Key}", kv.Value);
        }

        return result;
    }

    private AxolValue EvalImport(ListForm form, Environment env)
    {
        if (form.Args.Count < 1)
            throw new AxolRuntimeException("import: expected (import moduleName) or (import \"file.axol\")");

        if (form.Args[0] is StringLit filePath)
        {
            // File-based import
            var source = File.ReadAllText(filePath.Value);
            var lexer = new Axol.Lexer.AxolLexer(source, filePath.Value);
            var tokens = lexer.Tokenize();
            var parser = new Axol.Parser.AxolParser(tokens, _diagnostics);
            var program = parser.ParseProgram();

            if (_diagnostics.HasErrors)
                throw new AxolRuntimeException($"import: parse errors in {filePath.Value}");

            // Execute in isolated environment
            var fileEnv = _global.CreateChild();
            Eval(program, fileEnv);

            // Export bindings to current scope
            foreach (var kv in fileEnv.GetAllBindings())
            {
                if (!_global.TryGet(kv.Key, out _))
                    env.Define(kv.Key, kv.Value);
            }

            return UnitVal.Instance;
        }

        if (form.Args[0] is SymbolRef modRef)
        {
            // Module-based import: make ModName.xxx accessible
            if (!_modules.TryGet(modRef.Name, out _))
                throw new AxolRuntimeException($"import: module '{modRef.Name}' not found");
            // Already accessible via ModName.xxx from EvalModule
            return UnitVal.Instance;
        }

        throw new AxolRuntimeException("import: expected module name or file path");
    }

    private AxolValue EvalUse(ListForm form, Environment env)
    {
        // (use math.square) → import math.square as square
        if (form.Args.Count < 1)
            throw new AxolRuntimeException("use: expected (use module.binding)");

        string? qualName = null;
        if (form.Args[0] is SymbolRef useRef)
            qualName = useRef.Name;
        else
            qualName = TryReconstructDottedName(form.Args[0]);

        if (qualName == null)
            throw new AxolRuntimeException("use: expected (use module.binding)");

        var dotIdx = qualName.IndexOf('.');
        if (dotIdx < 0)
            throw new AxolRuntimeException("use: expected qualified name like module.name");

        var shortName = qualName[(dotIdx + 1)..];

        // Look up the qualified binding
        if (env.TryGet(qualName, out var val))
        {
            env.Define(shortName, val);
            return UnitVal.Instance;
        }

        throw new AxolRuntimeException($"use: '{qualName}' not found");
    }

    // ─── Arithmetic / Comparison / Logic ───

    private AxolValue EvalArithmetic(ListForm form, Environment env)
    {
        var op = form.Keyword;
        if (form.Args.Count == 0)
            throw new AxolRuntimeException($"{op}: expected operands");

        if (op == "-" && form.Args.Count == 1)
        {
            var val = Eval(form.Args[0], env);
            return val switch
            {
                IntVal iv => new IntVal(-iv.Value),
                FloatVal fv => new FloatVal(-fv.Value),
                _ => throw new AxolRuntimeException($"-: expected number")
            };
        }

        var left = Eval(form.Args[0], env);
        var right = Eval(form.Args[1], env);

        if (op == "+" && left is StrVal ls && right is StrVal rs)
            return new StrVal(ls.Value + rs.Value);

        bool useFloat = left is FloatVal || right is FloatVal;

        if (useFloat)
        {
            var a = Builtins.ToDouble(left);
            var b = Builtins.ToDouble(right);
            return new FloatVal(op switch
            {
                "+" => a + b,
                "-" => a - b,
                "*" => a * b,
                "/" => b != 0 ? a / b : throw new AxolRuntimeException("Division by zero"),
                "%" => b != 0 ? a % b : throw new AxolRuntimeException("Division by zero"),
                _ => throw new AxolRuntimeException($"Unknown op {op}")
            });
        }
        else
        {
            var a = Builtins.ToLong(left);
            var b = Builtins.ToLong(right);
            return new IntVal(op switch
            {
                "+" => a + b,
                "-" => a - b,
                "*" => a * b,
                "/" => b != 0 ? a / b : throw new AxolRuntimeException("Division by zero"),
                "%" => b != 0 ? a % b : throw new AxolRuntimeException("Division by zero"),
                _ => throw new AxolRuntimeException($"Unknown op {op}")
            });
        }
    }

    private AxolValue EvalComparison(ListForm form, Environment env)
    {
        if (form.Args.Count < 2)
            throw new AxolRuntimeException($"{form.Keyword}: expected 2 operands");

        var left = Eval(form.Args[0], env);
        var right = Eval(form.Args[1], env);

        if (form.Keyword == "=" || form.Keyword == "!=")
        {
            var eq = Builtins.ValuesEqual(left, right);
            return new BoolVal(form.Keyword == "=" ? eq : !eq);
        }

        var a = Builtins.ToDouble(left);
        var b = Builtins.ToDouble(right);
        return new BoolVal(form.Keyword switch
        {
            "<" => a < b,
            ">" => a > b,
            "<=" => a <= b,
            ">=" => a >= b,
            _ => throw new AxolRuntimeException($"Unknown comparison: {form.Keyword}")
        });
    }

    private AxolValue EvalLogical(ListForm form, Environment env)
    {
        return form.Keyword switch
        {
            "&" => new BoolVal(Eval(form.Args[0], env).IsTruthy && Eval(form.Args[1], env).IsTruthy),
            "|" => new BoolVal(Eval(form.Args[0], env).IsTruthy || Eval(form.Args[1], env).IsTruthy),
            "~" => new BoolVal(!Eval(form.Args[0], env).IsTruthy),
            _ => throw new AxolRuntimeException($"Unknown logical op: {form.Keyword}")
        };
    }

    private AxolValue EvalFunctionCall(ListForm form, Environment env)
    {
        var fn = env.Get(form.Keyword);
        var args = form.Args.Select(a => Eval(a, env)).ToList();
        return CallFunction(fn, args);
    }

    public AxolValue CallFunction(AxolValue fn, IReadOnlyList<AxolValue> args)
    {
        if (fn is BuiltinFunctionVal builtin)
            return builtin.Impl(args);

        if (fn is FunctionVal func)
        {
            var callEnv = func.Closure.CreateChild();
            for (int i = 0; i < func.Params.Count && i < args.Count; i++)
                callEnv.Define(func.Params[i], args[i]);

            try
            {
                AxolValue result = UnitVal.Instance;
                foreach (var bodyNode in func.Body)
                    result = Eval(bodyNode, callEnv);
                return result;
            }
            catch (ReturnSignal ret)
            {
                return ret.Value;
            }
        }

        throw new AxolRuntimeException($"Not a function: {fn.Display()}");
    }

    private static string FormatNode(AstNode node) => node switch
    {
        ListForm lf => $"({lf.Keyword} {string.Join(" ", lf.Args.Select(FormatNode))})",
        SymbolRef sr => sr.Name,
        NumberLitInt ni => ni.Value.ToString(),
        NumberLitFloat nf => nf.Value.ToString(System.Globalization.CultureInfo.InvariantCulture),
        StringLit sl => $"\"{sl.Value}\"",
        BoolLit bl => bl.Value ? "true" : "false",
        NilLit => "nil",
        _ => "?"
    };

    private static string EscapeJson(string s) => s.Replace("\\", "\\\\").Replace("\"", "\\\"");
}
