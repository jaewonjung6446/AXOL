using Axol.Core.Ast;
using Axol.Core.Diagnostics;

namespace Axol.TypeChecker;

public sealed class TypeInference
{
    private readonly Unifier _unifier = new();
    private readonly TypeEnvironment _env;
    private readonly DiagnosticBag _diagnostics;

    public TypeInference(TypeEnvironment? env = null, DiagnosticBag? diagnostics = null)
    {
        _env = env ?? new TypeEnvironment();
        _diagnostics = diagnostics ?? new DiagnosticBag();
        BuiltinTypes.Register(_env);
    }

    public AxolType Infer(AstNode node) => InferNode(node, _env);

    private AxolType InferNode(AstNode node, TypeEnvironment env)
    {
        return node switch
        {
            NumberLitInt => IntType.Instance,
            NumberLitFloat => FloatType.Instance,
            StringLit => StringType.Instance,
            BoolLit => BoolType.Instance,
            NilLit => NilType.Instance,
            SymbolRef sym => InferSymbol(sym, env),
            ListForm form => InferListForm(form, env),
            TypeAnnotation => UnitType.Instance,
            Program p => InferProgram(p, env),
            _ => UnknownType.Instance
        };
    }

    private AxolType InferSymbol(SymbolRef sym, TypeEnvironment env)
    {
        var t = env.Lookup(sym.Name);
        if (t == null)
            return UnknownType.Instance;

        if (t is ForAllType scheme)
            return _unifier.Instantiate(scheme);

        return t;
    }

    private AxolType InferProgram(Program p, TypeEnvironment env)
    {
        AxolType result = UnitType.Instance;
        foreach (var form in p.Forms)
            result = InferNode(form, env);
        return result;
    }

    private AxolType InferListForm(ListForm form, TypeEnvironment env)
    {
        return form.Keyword switch
        {
            "f" => InferFuncDef(form, env),
            "v" or "m" => InferLetBinding(form, env),
            "?" => InferIf(form, env),
            "+" or "-" or "*" or "/" or "%" => InferArithmetic(form, env),
            "=" or "!=" or "<" or ">" or "<=" or ">=" => BoolType.Instance,
            "&" or "|" or "~" => BoolType.Instance,
            "A" => InferArrayLiteral(form, env),
            "H" => InferMapLiteral(form, env),
            "S" => InferStructLiteral(form, env),
            "@" => InferFieldAccess(form, env),
            "#" => InferIndexAccess(form, env),
            "D" => InferDo(form, env),
            "L" => InferLambda(form, env),
            "X" => InferMatch(form, env),
            "W" or "F" => UnitType.Instance,
            "R" => form.Args.Count > 0 ? InferNode(form.Args[0], env) : UnitType.Instance,
            "!" or "Q" or "G" => UnitType.Instance,
            "E" => UnknownType.Instance, // throw never returns normally
            "C" => InferCatch(form, env),
            "t" or "e" => UnitType.Instance,
            "M" => InferModule(form, env),
            "import" or "use" => UnitType.Instance,
            "m!" => InferMutate(form, env),
            "P" => InferPipe(form, env),
            _ => InferFunctionCall(form, env)
        };
    }

    private AxolType InferFuncDef(ListForm form, TypeEnvironment env)
    {
        int idx = 0;
        var args = form.Args;
        if (idx >= args.Count || args[idx] is not SymbolRef nameRef) return UnknownType.Instance;
        idx++;

        // Parse type annotation if present
        AxolType? annotatedType = null;
        if (idx < args.Count && args[idx] is TypeAnnotation ta)
        {
            annotatedType = TypeAnnotationToType(ta);
            idx++;
        }

        // Collect params
        var paramNames = new List<string>();
        while (idx < args.Count && args[idx] is SymbolRef pRef && IsParam(args, idx))
        {
            paramNames.Add(pRef.Name);
            idx++;
        }

        // Create function type
        var paramTypes = new List<AxolType>();
        var bodyEnv = env.CreateChild();

        if (annotatedType is FnType annotatedFn)
        {
            for (int i = 0; i < paramNames.Count; i++)
            {
                var pt = i < annotatedFn.Params.Count ? annotatedFn.Params[i] : _unifier.FreshVar();
                paramTypes.Add(pt);
                bodyEnv.Define(paramNames[i], pt);
            }
        }
        else
        {
            foreach (var pName in paramNames)
            {
                var pt = _unifier.FreshVar();
                paramTypes.Add(pt);
                bodyEnv.Define(pName, pt);
            }
        }

        // Infer body
        AxolType bodyType = UnitType.Instance;
        for (int i = idx; i < args.Count; i++)
            bodyType = InferNode(args[i], bodyEnv);

        var fnType = new FnType(paramTypes, bodyType);

        // Validate against annotation
        if (annotatedType != null && !_unifier.Unify(fnType, annotatedType))
        {
            _diagnostics.ReportError(form.Span, "T100",
                $"Function '{nameRef.Name}' type mismatch: inferred {FormatType(_unifier.Apply(fnType))}, annotated {FormatType(annotatedType)}");
        }

        var resultType = _unifier.Apply(fnType);

        // Generalize and store
        var scheme = _unifier.Generalize(resultType, env.FreeTypeVars());
        env.Define(nameRef.Name, scheme);

        return resultType;
    }

    private static bool IsParam(IReadOnlyList<AstNode> args, int idx)
    {
        if (args[idx] is ListForm) return false;
        if (args[idx] is NumberLitInt or NumberLitFloat or StringLit or BoolLit or NilLit) return false;
        if (idx == args.Count - 1 && args[idx] is SymbolRef) return false;
        return true;
    }

    private AxolType InferLetBinding(ListForm form, TypeEnvironment env)
    {
        if (form.Args.Count < 2 || form.Args[0] is not SymbolRef nameRef)
            return UnknownType.Instance;

        var valType = InferNode(form.Args[1], env);
        var scheme = _unifier.Generalize(valType, env.FreeTypeVars());
        env.Define(nameRef.Name, scheme);
        return valType;
    }

    private AxolType InferIf(ListForm form, TypeEnvironment env)
    {
        if (form.Args.Count < 2) return UnknownType.Instance;

        var condType = InferNode(form.Args[0], env);
        if (!_unifier.Unify(condType, BoolType.Instance))
        {
            _diagnostics.ReportError(form.Span, "T101", "Condition must be boolean");
        }

        var thenType = InferNode(form.Args[1], env);
        if (form.Args.Count >= 3)
        {
            var elseType = InferNode(form.Args[2], env);
            if (!_unifier.Unify(thenType, elseType))
            {
                _diagnostics.ReportError(form.Span, "T102",
                    $"If branches have different types: {FormatType(_unifier.Apply(thenType))} vs {FormatType(_unifier.Apply(elseType))}");
            }
        }
        return _unifier.Apply(thenType);
    }

    private AxolType InferArithmetic(ListForm form, TypeEnvironment env)
    {
        if (form.Args.Count == 0) return UnknownType.Instance;

        if (form.Keyword == "-" && form.Args.Count == 1)
            return InferNode(form.Args[0], env);

        var leftType = InferNode(form.Args[0], env);
        var rightType = form.Args.Count > 1 ? InferNode(form.Args[1], env) : leftType;

        leftType = _unifier.Apply(leftType);
        rightType = _unifier.Apply(rightType);

        // String concat with +
        if (form.Keyword == "+" && leftType is StringType && rightType is StringType)
            return StringType.Instance;

        // Float promotion
        if (leftType is FloatType || rightType is FloatType)
            return FloatType.Instance;

        return IntType.Instance;
    }

    private AxolType InferArrayLiteral(ListForm form, TypeEnvironment env)
    {
        if (form.Args.Count == 0)
            return new ListType(_unifier.FreshVar());

        var elemType = InferNode(form.Args[0], env);
        for (int i = 1; i < form.Args.Count; i++)
        {
            var t = InferNode(form.Args[i], env);
            _unifier.Unify(elemType, t);
        }
        return new ListType(_unifier.Apply(elemType));
    }

    private AxolType InferMapLiteral(ListForm form, TypeEnvironment env)
    {
        if (form.Args.Count < 2)
            return new MapType(StringType.Instance, _unifier.FreshVar());

        var keyType = InferNode(form.Args[0], env);
        var valType = InferNode(form.Args[1], env);

        for (int i = 2; i + 1 < form.Args.Count; i += 2)
        {
            _unifier.Unify(keyType, InferNode(form.Args[i], env));
            _unifier.Unify(valType, InferNode(form.Args[i + 1], env));
        }

        return new MapType(_unifier.Apply(keyType), _unifier.Apply(valType));
    }

    private AxolType InferStructLiteral(ListForm form, TypeEnvironment env)
    {
        if (form.Args.Count < 1 || form.Args[0] is not SymbolRef typeRef)
            return UnknownType.Instance;
        return new NamedType(typeRef.Name);
    }

    private AxolType InferFieldAccess(ListForm form, TypeEnvironment env)
    {
        if (form.Args.Count < 2) return UnknownType.Instance;
        InferNode(form.Args[0], env);
        return _unifier.FreshVar(); // We don't know field types statically
    }

    private AxolType InferIndexAccess(ListForm form, TypeEnvironment env)
    {
        if (form.Args.Count < 2) return UnknownType.Instance;
        var collType = InferNode(form.Args[0], env);
        collType = _unifier.Apply(collType);

        if (collType is ListType lt)
            return lt.Element;
        if (collType is StringType)
            return StringType.Instance;

        return _unifier.FreshVar();
    }

    private AxolType InferDo(ListForm form, TypeEnvironment env)
    {
        var doEnv = env.CreateChild();
        AxolType result = UnitType.Instance;
        foreach (var arg in form.Args)
            result = InferNode(arg, doEnv);
        return result;
    }

    private AxolType InferLambda(ListForm form, TypeEnvironment env)
    {
        var paramNames = new List<string>();
        int bodyStart;

        if (form.Args[0] is ListForm paramList)
        {
            foreach (var p in paramList.Args)
                if (p is SymbolRef sr) paramNames.Add(sr.Name);
            bodyStart = 1;
        }
        else if (form.Args[0] is SymbolRef sp)
        {
            paramNames.Add(sp.Name);
            bodyStart = 1;
        }
        else
        {
            bodyStart = 0;
        }

        var bodyEnv = env.CreateChild();
        var paramTypes = new List<AxolType>();
        foreach (var pName in paramNames)
        {
            var pt = _unifier.FreshVar();
            paramTypes.Add(pt);
            bodyEnv.Define(pName, pt);
        }

        AxolType bodyType = UnitType.Instance;
        for (int i = bodyStart; i < form.Args.Count; i++)
            bodyType = InferNode(form.Args[i], bodyEnv);

        return new FnType(paramTypes.Select(p => _unifier.Apply(p)).ToList(), _unifier.Apply(bodyType));
    }

    private AxolType InferMatch(ListForm form, TypeEnvironment env)
    {
        if (form.Args.Count < 3) return UnknownType.Instance;
        InferNode(form.Args[0], env);

        // Result type from first arm body
        var resultType = _unifier.FreshVar();
        int i = 1;
        while (i + 1 < form.Args.Count)
        {
            i++; // skip pattern
            if (i < form.Args.Count && form.Args[i] is ListForm gf && gf.Keyword == "when")
                i++; // skip guard
            if (i < form.Args.Count)
            {
                var armType = InferNode(form.Args[i], env);
                _unifier.Unify(resultType, armType);
                i++;
            }
        }
        return _unifier.Apply(resultType);
    }

    private AxolType InferCatch(ListForm form, TypeEnvironment env)
    {
        if (form.Args.Count < 3) return UnknownType.Instance;
        var tryType = InferNode(form.Args[0], env);
        // Handler has string error variable
        var catchEnv = env.CreateChild();
        if (form.Args[1] is SymbolRef errRef)
            catchEnv.Define(errRef.Name, StringType.Instance);
        var handlerType = InferNode(form.Args[2], catchEnv);
        _unifier.Unify(tryType, handlerType);
        return _unifier.Apply(tryType);
    }

    private AxolType InferMutate(ListForm form, TypeEnvironment env)
    {
        if (form.Args.Count < 2) return UnknownType.Instance;
        return InferNode(form.Args[1], env);
    }

    private AxolType InferPipe(ListForm form, TypeEnvironment env)
    {
        if (form.Args.Count < 2) return UnknownType.Instance;
        var valType = InferNode(form.Args[0], env);
        for (int i = 1; i < form.Args.Count; i++)
        {
            var fnType = InferNode(form.Args[i], env);
            fnType = _unifier.Apply(fnType);
            if (fnType is FnType ft && ft.Params.Count > 0)
            {
                _unifier.Unify(valType, ft.Params[0]);
                valType = ft.Return;
            }
            else
            {
                valType = _unifier.FreshVar();
            }
        }
        return _unifier.Apply(valType);
    }

    private AxolType InferModule(ListForm form, TypeEnvironment env)
    {
        if (form.Args.Count < 1) return UnitType.Instance;
        var modEnv = env.CreateChild();
        AxolType result = UnitType.Instance;
        for (int i = 1; i < form.Args.Count; i++)
            result = InferNode(form.Args[i], modEnv);
        return result;
    }

    private AxolType InferFunctionCall(ListForm form, TypeEnvironment env)
    {
        var fnType = env.Lookup(form.Keyword);
        if (fnType == null)
            return UnknownType.Instance;

        if (fnType is ForAllType scheme)
            fnType = _unifier.Instantiate(scheme);

        fnType = _unifier.Apply(fnType);

        if (fnType is FnType ft)
        {
            // Infer arg types and unify with params
            for (int i = 0; i < form.Args.Count && i < ft.Params.Count; i++)
            {
                var argType = InferNode(form.Args[i], env);
                if (!_unifier.Unify(ft.Params[i], argType))
                {
                    _diagnostics.ReportError(form.Args[i].Span, "T103",
                        $"Argument type mismatch in '{form.Keyword}': expected {FormatType(_unifier.Apply(ft.Params[i]))}, got {FormatType(_unifier.Apply(argType))}");
                }
            }
            return _unifier.Apply(ft.Return);
        }

        return UnknownType.Instance;
    }

    private AxolType TypeAnnotationToType(TypeAnnotation ta)
    {
        var paramTypes = ta.Types.Select(TypeNodeToType).ToList();
        var retType = ta.ReturnType != null ? TypeNodeToType(ta.ReturnType) : UnitType.Instance;

        if (paramTypes.Count == 0 && ta.ReturnType == null)
            return UnitType.Instance;

        return new FnType(paramTypes, retType);
    }

    private static AxolType TypeNodeToType(Axol.Core.Ast.TypeNode tn) => tn switch
    {
        Axol.Core.Ast.PrimitiveType pt => pt.Name switch
        {
            "i" => IntType.Instance,
            "f" => FloatType.Instance,
            "b" => BoolType.Instance,
            "s" => StringType.Instance,
            "n" => NilType.Instance,
            "u" => UnitType.Instance,
            _ => UnknownType.Instance
        },
        Axol.Core.Ast.ArrayType at => new ListType(TypeNodeToType(at.Element)),
        Axol.Core.Ast.MapType mt => new MapType(TypeNodeToType(mt.Key), TypeNodeToType(mt.ValueType)),
        Axol.Core.Ast.OptionalType ot => new OptionalType(TypeNodeToType(ot.Inner)),
        Axol.Core.Ast.FunctionType ft => new FnType(ft.Params.Select(TypeNodeToType).ToList(), TypeNodeToType(ft.Return)),
        Axol.Core.Ast.NamedType nt => new NamedType(nt.Name),
        _ => UnknownType.Instance
    };

    private static string FormatType(AxolType t) => t switch
    {
        IntType => "Int",
        FloatType => "Float",
        BoolType => "Bool",
        StringType => "String",
        UnitType => "Unit",
        NilType => "Nil",
        ListType lt => $"[{FormatType(lt.Element)}]",
        MapType mt => $"{{{FormatType(mt.Key)}:{FormatType(mt.Value)}}}",
        FnType fn => $"({string.Join(", ", fn.Params.Select(FormatType))} -> {FormatType(fn.Return)})",
        TypeVar tv => $"t{tv.Id}",
        NamedType nt => nt.Name,
        OptionalType ot => $"?{FormatType(ot.Inner)}",
        ForAllType fa => $"∀{string.Join(",", fa.Vars)}.{FormatType(fa.Body)}",
        ErrorType et => $"Error({et.Message})",
        UnknownType => "?",
        _ => "?"
    };
}
