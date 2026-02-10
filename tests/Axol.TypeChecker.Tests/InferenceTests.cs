using Axol.Core.Ast;
using Axol.Core.Diagnostics;
using Axol.Lexer;
using Axol.Parser;
using Axol.TypeChecker;
using Xunit;

namespace Axol.TypeChecker.Tests;

public class InferenceTests
{
    private static TypeInference CreateInference(DiagnosticBag? diag = null)
    {
        return new TypeInference(null, diag);
    }

    private static AstNode ParseExpr(string source)
    {
        var lexer = new AxolLexer(source);
        var tokens = lexer.Tokenize();
        var parser = new AxolParser(tokens);
        var prog = parser.ParseProgram();
        return prog;
    }

    private static AxolType InferType(string source)
    {
        var inf = CreateInference();
        var node = ParseExpr(source);
        return inf.Infer(node);
    }

    private static (AxolType type, DiagnosticBag diag) InferWithDiag(string source)
    {
        var diag = new DiagnosticBag();
        var inf = new TypeInference(null, diag);
        var node = ParseExpr(source);
        var type = inf.Infer(node);
        return (type, diag);
    }

    // --- Basic literal inference ---

    [Fact]
    public void InferInt()
    {
        Assert.IsType<IntType>(InferType("42"));
    }

    [Fact]
    public void InferFloat()
    {
        Assert.IsType<FloatType>(InferType("3.14"));
    }

    [Fact]
    public void InferString()
    {
        Assert.IsType<StringType>(InferType("\"hello\""));
    }

    [Fact]
    public void InferBool()
    {
        Assert.IsType<BoolType>(InferType("true"));
    }

    [Fact]
    public void InferNil()
    {
        Assert.IsType<NilType>(InferType("nil"));
    }

    // --- Arithmetic inference ---

    [Fact]
    public void InferAdd_IntInt()
    {
        Assert.IsType<IntType>(InferType("(+ 1 2)"));
    }

    [Fact]
    public void InferAdd_FloatPromotion()
    {
        Assert.IsType<FloatType>(InferType("(+ 1 2.0)"));
    }

    [Fact]
    public void InferAdd_StringConcat()
    {
        Assert.IsType<StringType>(InferType("(+ \"a\" \"b\")"));
    }

    [Fact]
    public void InferSub()
    {
        Assert.IsType<IntType>(InferType("(- 5 3)"));
    }

    [Fact]
    public void InferMul()
    {
        Assert.IsType<IntType>(InferType("(* 2 3)"));
    }

    [Fact]
    public void InferDiv_Float()
    {
        Assert.IsType<FloatType>(InferType("(/ 1.0 2.0)"));
    }

    // --- Comparison ---

    [Fact]
    public void InferComparison()
    {
        Assert.IsType<BoolType>(InferType("(> 1 2)"));
    }

    [Fact]
    public void InferEquality()
    {
        Assert.IsType<BoolType>(InferType("(= 1 1)"));
    }

    [Fact]
    public void InferLogical()
    {
        Assert.IsType<BoolType>(InferType("(& true false)"));
    }

    // --- Variable binding ---

    [Fact]
    public void InferLetBinding()
    {
        Assert.IsType<IntType>(InferType("(v x 42) x"));
    }

    [Fact]
    public void InferMutableBinding()
    {
        Assert.IsType<IntType>(InferType("(m x 42) x"));
    }

    // --- Conditional ---

    [Fact]
    public void InferIf_SameBranches()
    {
        Assert.IsType<IntType>(InferType("(? true 1 2)"));
    }

    [Fact]
    public void InferIf_NonBoolCondition_Error()
    {
        var (_, diag) = InferWithDiag("(? 42 1 2)");
        Assert.True(diag.HasErrors);
        Assert.Contains(diag.All, d => d.Code == "T101");
    }

    [Fact]
    public void InferIf_DifferentBranches_Error()
    {
        var (_, diag) = InferWithDiag("(? true 1 \"hello\")");
        Assert.True(diag.HasErrors);
        Assert.Contains(diag.All, d => d.Code == "T102");
    }

    // --- Function definition ---

    [Fact]
    public void InferFuncDef()
    {
        var ty = InferType("(f double x (* x 2))");
        Assert.IsType<FnType>(ty);
    }

    [Fact]
    public void InferFuncDef_WithAnnotation()
    {
        var ty = InferType("(f double [i -> i] x (* x 2))");
        var fn = Assert.IsType<FnType>(ty);
        Assert.Single(fn.Params);
    }

    [Fact]
    public void InferFuncDef_AnnotationMismatch_Error()
    {
        var (_, diag) = InferWithDiag("(f wrong [s -> s] x (+ x 1))");
        // This should report a type mismatch error since body returns Int but annotation says String
        Assert.True(diag.HasErrors);
    }

    [Fact]
    public void InferFuncCall()
    {
        Assert.IsType<IntType>(InferType("(f double [i -> i] x (* x 2)) (double 5)"));
    }

    // --- Array/Collection inference ---

    [Fact]
    public void InferArrayLiteral_Int()
    {
        var ty = InferType("(A 1 2 3)");
        var list = Assert.IsType<ListType>(ty);
        Assert.IsType<IntType>(list.Element);
    }

    [Fact]
    public void InferArrayLiteral_String()
    {
        var ty = InferType("(A \"a\" \"b\")");
        var list = Assert.IsType<ListType>(ty);
        Assert.IsType<StringType>(list.Element);
    }

    [Fact]
    public void InferArrayLiteral_Empty()
    {
        var ty = InferType("(A)");
        Assert.IsType<ListType>(ty);
    }

    [Fact]
    public void InferMapLiteral()
    {
        var ty = InferType("(H \"a\" 1 \"b\" 2)");
        var map = Assert.IsType<MapType>(ty);
        Assert.IsType<StringType>(map.Key);
        Assert.IsType<IntType>(map.Value);
    }

    // --- Struct ---

    [Fact]
    public void InferStructLiteral()
    {
        var ty = InferType("(S Point x 10 y 20)");
        Assert.IsType<NamedType>(ty);
    }

    // --- Lambda ---

    [Fact]
    public void InferLambda()
    {
        var ty = InferType("(L x (+ x 1))");
        Assert.IsType<FnType>(ty);
    }

    // --- Pipe ---

    [Fact]
    public void InferPipe()
    {
        var ty = InferType("(f inc [i -> i] x (+ x 1)) (f dbl [i -> i] x (* x 2)) (P 3 inc dbl)");
        Assert.IsType<IntType>(ty);
    }

    // --- Do block ---

    [Fact]
    public void InferDoBlock()
    {
        Assert.IsType<IntType>(InferType("(D (v x 1) (+ x 2))"));
    }

    // --- Builtin function call ---

    [Fact]
    public void InferBuiltinCall_Len()
    {
        Assert.IsType<IntType>(InferType("(len (A 1 2 3))"));
    }

    [Fact]
    public void InferBuiltinCall_Str()
    {
        Assert.IsType<StringType>(InferType("(str 42)"));
    }

    [Fact]
    public void InferBuiltinCall_Sqrt()
    {
        Assert.IsType<FloatType>(InferType("(sqrt 4.0)"));
    }

    // --- Let-polymorphism ---

    [Fact]
    public void LetPolymorphism_IdentityUsedMultipleTypes()
    {
        var (_, diag) = InferWithDiag("(f id x x) (id 42) (id \"hello\")");
        Assert.False(diag.HasErrors);
    }

    // --- Return ---

    [Fact]
    public void InferReturn()
    {
        var ty = InferType("(R 42)");
        Assert.IsType<IntType>(ty);
    }

    // --- Match ---

    [Fact]
    public void InferMatch()
    {
        var ty = InferType("(X 1 1 \"one\" 2 \"two\" _ \"other\")");
        Assert.IsType<StringType>(ty);
    }

    // --- Catch ---

    [Fact]
    public void InferCatch()
    {
        var ty = InferType("(C 42 err 0)");
        Assert.IsType<IntType>(ty);
    }

    // --- Module ---

    [Fact]
    public void InferModule_NoError()
    {
        var (_, diag) = InferWithDiag("(M math (f square x (* x x)))");
        Assert.False(diag.HasErrors);
    }

    // --- TypeChecker integration ---

    [Fact]
    public void TypeChecker_ValidProgram_NoErrors()
    {
        var lexer = new AxolLexer("(f id [i -> i] x x)");
        var tokens = lexer.Tokenize();
        var diag = new DiagnosticBag();
        var parser = new AxolParser(tokens, diag);
        var prog = parser.ParseProgram();
        var checker = new AxolTypeChecker(diag);
        checker.Check(prog);
        Assert.False(diag.HasErrors);
    }

    [Fact]
    public void TypeChecker_ComplexProgram_NoErrors()
    {
        var source = "(f fib [i -> i] n (? (<= n 1) n (+ (fib (- n 1)) (fib (- n 2)))))";
        var lexer = new AxolLexer(source);
        var tokens = lexer.Tokenize();
        var diag = new DiagnosticBag();
        var parser = new AxolParser(tokens, diag);
        var prog = parser.ParseProgram();
        var checker = new AxolTypeChecker(diag);
        checker.Check(prog);
        Assert.False(diag.HasErrors);
    }

    // --- Unary minus ---

    [Fact]
    public void InferUnaryMinus()
    {
        var ty = InferType("(- 5)");
        Assert.IsType<IntType>(ty);
    }

    // --- Nested expressions ---

    [Fact]
    public void InferNestedArithmetic()
    {
        Assert.IsType<IntType>(InferType("(+ (* 2 3) (- 10 4))"));
    }

    // --- Error type validation ---

    [Fact]
    public void InferArgTypeMismatch_Error()
    {
        var (_, diag) = InferWithDiag("(f add [i i -> i] a b (+ a b)) (add \"hello\" \"world\")");
        Assert.True(diag.HasErrors);
        Assert.Contains(diag.All, d => d.Code == "T103");
    }

    // --- ForAllType ---

    [Fact]
    public void ForAllType_HasVars()
    {
        var vars = new[] { 1, 2 };
        var body = new FnType(new AxolType[] { new TypeVar(1) }, new TypeVar(2));
        var fa = new ForAllType(vars, body);
        Assert.Equal(2, fa.Vars.Length);
    }

    // --- ErrorType sentinel ---

    [Fact]
    public void ErrorType_Stores_Message()
    {
        var err = new ErrorType("test error");
        Assert.Equal("test error", err.Message);
    }
}
