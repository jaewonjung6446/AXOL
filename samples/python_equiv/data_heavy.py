class Unit:
    def __init__(self, name, hp, atk, defense):
        self.name = name
        self.hp = hp
        self.atk = atk
        self.defense = defense

class Skill:
    def __init__(self, dmg, cost):
        self.dmg = dmg
        self.cost = cost

heroes = [
    Unit("Knight", 120, 30, 20),
    Unit("Mage", 80, 50, 10),
    Unit("Rogue", 90, 40, 15),
]

skills = {
    "slash": Skill(25, 10),
    "fireball": Skill(50, 30),
    "backstab": Skill(35, 15),
}

def total_hp(units):
    s = 0
    for u in units:
        s += u.hp
    return s

def strongest(units):
    best = units[0]
    for u in units:
        if u.atk > best.atk:
            best = u
    return best

print("Total HP:", total_hp(heroes))
print("Strongest:", strongest(heroes).name)
