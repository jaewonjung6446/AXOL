using Axol.Core.Diagnostics;
using Axol.Interpreter;
using Axol.Interpreter.Values;
using Axol.Lexer;
using Axol.Parser;
using Xunit;

namespace Axol.Interpreter.Tests;

public class InterpreterTests
{
    private static (AxolValue result, string output) Run(string source)
    {
        var writer = new StringWriter();
        var lexer = new AxolLexer(source);
        var tokens = lexer.Tokenize();
        var diag = new DiagnosticBag();
        var parser = new AxolParser(tokens, diag);
        var program = parser.ParseProgram();
        Assert.False(diag.HasErrors, string.Join("\n", diag.All.Select(d => d.ToJson())));
        var interp = new AxolInterpreter(writer, diag);
        var result = interp.Run(program);
        return (result, writer.ToString().TrimEnd().Replace("\r\n", "\n"));
    }

    [Fact]
    public void Arithmetic_Add()
    {
        var (result, _) = Run("(+ 1 2)");
        Assert.Equal(3L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Arithmetic_Subtract()
    {
        var (result, _) = Run("(- 10 3)");
        Assert.Equal(7L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Arithmetic_Multiply()
    {
        var (result, _) = Run("(* 4 5)");
        Assert.Equal(20L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Arithmetic_Divide()
    {
        var (result, _) = Run("(/ 20 4)");
        Assert.Equal(5L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Arithmetic_Modulo()
    {
        var (result, _) = Run("(% 7 3)");
        Assert.Equal(1L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void FloatArithmetic()
    {
        var (result, _) = Run("(+ 1.5 2.5)");
        Assert.Equal(4.0, Assert.IsType<FloatVal>(result).Value);
    }

    [Fact]
    public void Comparison_Equal()
    {
        var (result, _) = Run("(= 1 1)");
        Assert.True(Assert.IsType<BoolVal>(result).Value);
    }

    [Fact]
    public void Comparison_NotEqual()
    {
        var (result, _) = Run("(!= 1 2)");
        Assert.True(Assert.IsType<BoolVal>(result).Value);
    }

    [Fact]
    public void Comparison_Less()
    {
        var (result, _) = Run("(< 1 2)");
        Assert.True(Assert.IsType<BoolVal>(result).Value);
    }

    [Fact]
    public void Logical_And()
    {
        var (result, _) = Run("(& true true)");
        Assert.True(Assert.IsType<BoolVal>(result).Value);
    }

    [Fact]
    public void Logical_Not()
    {
        var (result, _) = Run("(~ false)");
        Assert.True(Assert.IsType<BoolVal>(result).Value);
    }

    [Fact]
    public void VariableBinding_Immutable()
    {
        var (result, _) = Run("(v x 42) x");
        Assert.Equal(42L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void VariableBinding_Mutable()
    {
        var (result, _) = Run("(m x 1) (m! x 2) x");
        Assert.Equal(2L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void ImmutableReassign_Throws()
    {
        Assert.Throws<AxolRuntimeException>(() => Run("(v x 1) (m! x 2)"));
    }

    [Fact]
    public void Conditional_TrueBranch()
    {
        var (result, _) = Run("(? true 1 2)");
        Assert.Equal(1L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Conditional_FalseBranch()
    {
        var (result, _) = Run("(? false 1 2)");
        Assert.Equal(2L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void FunctionDef_And_Call()
    {
        var (result, _) = Run("(f double [i -> i] x (* x 2)) (double 5)");
        Assert.Equal(10L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Fibonacci()
    {
        var (result, _) = Run("(f fib [i -> i] n (? (<= n 1) n (+ (fib (- n 1)) (fib (- n 2))))) (fib 10)");
        Assert.Equal(55L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Print()
    {
        var (_, output) = Run("(print \"hello\" \"world\")");
        Assert.Equal("hello world", output);
    }

    [Fact]
    public void ForLoop()
    {
        var (_, output) = Run("(F i (A 1 2 3) (print i))");
        Assert.Equal("1\n2\n3", output);
    }

    [Fact]
    public void WhileLoop()
    {
        var (_, output) = Run("(m x 0) (W (< x 3) (print x) (m! x (+ x 1)))");
        Assert.Equal("0\n1\n2", output);
    }

    [Fact]
    public void ArrayLiteral()
    {
        var (result, _) = Run("(A 1 2 3)");
        var list = Assert.IsType<ListVal>(result);
        Assert.Equal(3, list.Items.Count);
    }

    [Fact]
    public void HashMapLiteral()
    {
        var (result, _) = Run("(H \"a\" 1 \"b\" 2)");
        var map = Assert.IsType<MapVal>(result);
        Assert.Equal(2, map.Entries.Count);
    }

    [Fact]
    public void StructLiteral()
    {
        var (result, _) = Run("(S Point x 10 y 20)");
        var sv = Assert.IsType<StructVal>(result);
        Assert.Equal("Point", sv.TypeName);
        Assert.Equal(10L, Assert.IsType<IntVal>(sv.Fields["x"]).Value);
    }

    [Fact]
    public void FieldAccess()
    {
        var (result, _) = Run("(v p (S Point x 10 y 20)) (@ p x)");
        Assert.Equal(10L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void IndexAccess()
    {
        var (result, _) = Run("(v arr (A 10 20 30)) (# arr 1)");
        Assert.Equal(20L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void DoBlock()
    {
        var (result, _) = Run("(D (v x 1) (v y 2) (+ x y))");
        Assert.Equal(3L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Lambda()
    {
        var (result, _) = Run("(v dbl (L x (* x 2))) (dbl 5)");
        Assert.Equal(10L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Pipe()
    {
        var (result, _) = Run("(f inc [i -> i] x (+ x 1)) (f dbl [i -> i] x (* x 2)) (P 3 inc dbl)");
        Assert.Equal(8L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Assert_Pass()
    {
        var (result, _) = Run("(! (= 1 1))");
        Assert.IsType<UnitVal>(result);
    }

    [Fact]
    public void Assert_Fail()
    {
        Assert.Throws<AssertionFailedException>(() => Run("(! (= 1 2))"));
    }

    [Fact]
    public void Precondition_Pass()
    {
        var (result, _) = Run("(Q (> 5 0))");
        Assert.IsType<UnitVal>(result);
    }

    [Fact]
    public void Precondition_Fail()
    {
        Assert.Throws<ContractViolationException>(() => Run("(Q (> 0 5))"));
    }

    [Fact]
    public void ErrorHandling()
    {
        var (result, _) = Run("(C (E \"oops\") err err)");
        Assert.Equal("oops", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void StringConcat()
    {
        var (result, _) = Run("(+ \"hello\" \" world\")");
        Assert.Equal("hello world", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void Builtin_Len()
    {
        var (result, _) = Run("(len (A 1 2 3))");
        Assert.Equal(3L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Builtin_Range()
    {
        var (result, _) = Run("(len (range 0 5))");
        Assert.Equal(5L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void PatternMatch_Literal()
    {
        var (result, _) = Run("(X 2 1 \"one\" 2 \"two\" 3 \"three\")");
        Assert.Equal("two", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void PatternMatch_Wildcard()
    {
        var (result, _) = Run("(X 99 1 \"one\" _ \"other\")");
        Assert.Equal("other", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void UnaryMinus()
    {
        var (result, _) = Run("(- 5)");
        Assert.Equal(-5L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Builtin_Max()
    {
        var (result, _) = Run("(max 3 7)");
        Assert.Equal(7L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Builtin_Min()
    {
        var (result, _) = Run("(min 3 7)");
        Assert.Equal(3L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Builtin_Abs()
    {
        var (result, _) = Run("(abs -5)");
        Assert.Equal(5L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Return_Early()
    {
        var (result, _) = Run("(f test [] (R 42) 99) (test)");
        Assert.Equal(42L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void FibonacciOutput()
    {
        var (_, output) = Run("(f fib [i -> i] n (? (<= n 1) n (+ (fib (- n 1)) (fib (- n 2))))) (F i (range 0 10) (print (fib i)))");
        var lines = output.Split('\n');
        Assert.Equal(new[] { "0", "1", "1", "2", "3", "5", "8", "13", "21", "34" }, lines);
    }
}
