using Axol.Core.Tokens;
using Axol.Lexer;
using Xunit;

namespace Axol.Lexer.Tests;

public class Phase2LexerTests
{
    private static List<Token> Lex(string source) => new AxolLexer(source).Tokenize();

    // --- Step 1: Comments ---

    [Fact]
    public void LineComment_Ignored()
    {
        var tokens = Lex("; this is a comment\n42");
        Assert.Equal(TokenKind.IntLiteral, tokens[0].Kind);
        Assert.Equal(42L, tokens[0].Value);
    }

    [Fact]
    public void InlineComment_Ignored()
    {
        var tokens = Lex("(+ 1 2) ; add numbers");
        Assert.Equal(TokenKind.LParen, tokens[0].Kind);
        Assert.Equal(TokenKind.RParen, tokens[4].Kind);
        Assert.Equal(TokenKind.Eof, tokens[5].Kind);
    }

    [Fact]
    public void MultipleComments_Ignored()
    {
        var tokens = Lex("; line1\n; line2\n42");
        Assert.Equal(TokenKind.IntLiteral, tokens[0].Kind);
    }

    [Fact]
    public void CommentOnly_NoTokens()
    {
        var tokens = Lex("; just a comment");
        Assert.Single(tokens);
        Assert.Equal(TokenKind.Eof, tokens[0].Kind);
    }

    // --- Step 1: SourceMap ---

    [Fact]
    public void Source_Property_Exposed()
    {
        var lexer = new AxolLexer("(+ 1 2)");
        Assert.Equal("(+ 1 2)", lexer.Source);
    }

    // --- Step 2: Braces ---

    [Fact]
    public void Braces_Tokenized()
    {
        var tokens = Lex("{}");
        Assert.Equal(TokenKind.LBrace, tokens[0].Kind);
        Assert.Equal(TokenKind.RBrace, tokens[1].Kind);
        Assert.Equal(TokenKind.Eof, tokens[2].Kind);
    }

    [Fact]
    public void MapLiteral_Tokens()
    {
        var tokens = Lex("{\"a\" 1 \"b\" 2}");
        Assert.Equal(TokenKind.LBrace, tokens[0].Kind);
        Assert.Equal(TokenKind.StrLiteral, tokens[1].Kind);
        Assert.Equal(TokenKind.IntLiteral, tokens[2].Kind);
        Assert.Equal(TokenKind.RBrace, tokens[5].Kind);
    }

    [Fact]
    public void ArrayLiteral_Tokens()
    {
        var tokens = Lex("[1 2 3]");
        Assert.Equal(TokenKind.LBracket, tokens[0].Kind);
        Assert.Equal(TokenKind.IntLiteral, tokens[1].Kind);
        Assert.Equal(TokenKind.IntLiteral, tokens[3].Kind);
        Assert.Equal(TokenKind.RBracket, tokens[4].Kind);
    }

    // --- Step 2: Dot access ---

    [Fact]
    public void DotAccess_SingleSymbol()
    {
        var tokens = Lex("obj.field");
        Assert.Single(tokens, t => t.Kind == TokenKind.Symbol);
        Assert.Equal("obj.field", tokens[0].Lexeme);
    }

    [Fact]
    public void DotAccess_Chained()
    {
        var tokens = Lex("a.b.c");
        Assert.Equal(TokenKind.Symbol, tokens[0].Kind);
        Assert.Equal("a.b.c", tokens[0].Lexeme);
    }

    // --- Step 4: Rest pattern ---

    [Fact]
    public void RestPattern_Preserved()
    {
        var tokens = Lex("rest...");
        Assert.Equal(TokenKind.Symbol, tokens[0].Kind);
        Assert.Equal("rest...", tokens[0].Lexeme);
    }

    // --- Semicolon as delimiter ---

    [Fact]
    public void Semicolon_IsDelimiter()
    {
        var tokens = Lex("foo;comment");
        Assert.Equal(TokenKind.Symbol, tokens[0].Kind);
        Assert.Equal("foo", tokens[0].Lexeme);
        Assert.Equal(TokenKind.Eof, tokens[1].Kind);
    }
}
