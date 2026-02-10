using System;

class Entity
{
    public int Hp { get; set; }
    public int Atk { get; }
    public int Def { get; }

    public Entity(int hp, int atk, int def)
    {
        Hp = hp;
        Atk = atk;
        Def = def;
    }
}

class Program
{
    static Entity MakeEntity(string name, int hp, int atk, int def)
    {
        return new Entity(hp, atk, def);
    }

    static Entity ApplyDmg(Entity ent, int amt)
    {
        int oh = ent.Hp;
        int raw = amt - ent.Def;
        int dmg = raw < 0 ? 0 : raw;
        ent.Hp = Math.Max(0, oh - dmg);
        return ent;
    }

    static bool IsAlive(Entity ent) => ent.Hp > 0;

    static void Main()
    {
        var hero = MakeEntity("Hero", 100, 25, 10);
        var goblin = MakeEntity("Goblin", 30, 15, 5);

        goblin = ApplyDmg(goblin, hero.Atk);
        Console.WriteLine($"Goblin HP after hero attack: {goblin.Hp}");
        Console.WriteLine($"Goblin alive? {IsAlive(goblin)}");
    }
}
