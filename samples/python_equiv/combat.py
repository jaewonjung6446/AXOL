class Entity:
    def __init__(self, hp, atk, defense):
        self.hp = hp
        self.atk = atk
        self.defense = defense

def make_entity(name, hp, atk, defense):
    return Entity(hp, atk, defense)

def apply_dmg(ent, amt):
    assert amt >= 0, "precondition: amt >= 0"
    oh = ent.hp
    raw = amt - ent.defense
    dmg = 0 if raw < 0 else raw
    ent.hp = max(0, oh - dmg)
    assert ent.hp <= oh, "postcondition: hp <= old hp"
    return ent

def is_alive(ent):
    return ent.hp > 0

hero = make_entity("Hero", 100, 25, 10)
goblin = make_entity("Goblin", 30, 15, 5)

goblin = apply_dmg(goblin, hero.atk)
print("Goblin HP after hero attack:", goblin.hp)
print("Goblin alive?", is_alive(goblin))
