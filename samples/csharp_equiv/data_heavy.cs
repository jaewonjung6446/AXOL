using System;
using System.Collections.Generic;
using System.Linq;

class Unit
{
    public string Name { get; }
    public int Hp { get; }
    public int Atk { get; }
    public int Def { get; }

    public Unit(string name, int hp, int atk, int def)
    {
        Name = name;
        Hp = hp;
        Atk = atk;
        Def = def;
    }
}

class Skill
{
    public int Dmg { get; }
    public int Cost { get; }

    public Skill(int dmg, int cost)
    {
        Dmg = dmg;
        Cost = cost;
    }
}

class Program
{
    static int TotalHp(List<Unit> units) => units.Sum(u => u.Hp);

    static Unit Strongest(List<Unit> units) =>
        units.OrderByDescending(u => u.Atk).First();

    static void Main()
    {
        var heroes = new List<Unit>
        {
            new Unit("Knight", 120, 30, 20),
            new Unit("Mage", 80, 50, 10),
            new Unit("Rogue", 90, 40, 15),
        };

        var skills = new Dictionary<string, Skill>
        {
            ["slash"] = new Skill(25, 10),
            ["fireball"] = new Skill(50, 30),
            ["backstab"] = new Skill(35, 15),
        };

        Console.WriteLine($"Total HP: {TotalHp(heroes)}");
        Console.WriteLine($"Strongest: {Strongest(heroes).Name}");
    }
}
